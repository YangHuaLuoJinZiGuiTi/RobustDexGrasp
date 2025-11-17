import numpy as np
from qpsolvers import solve_qp
from copy import deepcopy


def normalize_vector(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def utils_1axis_to_3axes(
    axis_0, rot_base1=np.array([[0, 1, 0]]), rot_base2=np.array([[0, 0, 1]])
):
    """
    One 3D direction to three 3D axes for constructing 3x3 rotation matrix.

    Parameters
    ----------
    axis_0: np.array [n, 3]

    Returns
    ----------
    axis_0: np.array [n, 3]
    axis_1: np.array [n, 3]
    axis_2: np.array [n, 3]

    """

    proj_xy = np.abs(np.sum(axis_0 * rot_base1, axis=-1, keepdims=True))
    axis_1 = np.where(proj_xy > 0.99, rot_base2, rot_base1)

    axis_1 = normalize_vector(
        axis_1 - np.sum(axis_1 * axis_0, axis=-1, keepdims=True) * axis_0
    )
    axis_2 = np.cross(axis_0, axis_1, axis=-1)

    return axis_0, axis_1, axis_2


class ContactQP:
    def __init__(self, num_contact, miu_coef, solver_type):
        self.num_contact = num_contact
        self.miu_coef = miu_coef
        self.solver_type = solver_type

        self.G_matrix, self.h_matrix, self.E_matrix = self.init_LCQP()
        return

    def init_LCQP(self):
        """
        Build G matrix and h matrix for constraints Gx <= h,
        using soft contact model with pyramid discretization.

        """
        num_f_strength = self.num_contact * 6
        G_matrix = np.zeros((num_f_strength + self.num_contact + 1, num_f_strength))
        h_matrix = np.zeros((num_f_strength + self.num_contact + 1))

        # - force <= 0
        G_matrix[range(0, num_f_strength), range(0, num_f_strength)] = -1.0

        # pressure <= 1
        for i in range(self.num_contact):
            G_matrix[num_f_strength + i, 6 * i : 6 * i + 6] = 1.0
        h_matrix[-self.num_contact-1 :-1] = 1

        # - sum pressure <= -0.1 
        G_matrix[-1, :] = -1.0
        h_matrix[-1] = -0.1
        
        # https://mujoco.readthedocs.io/en/stable/_images/contact_frame.svg
        E_matrix = np.zeros((self.num_contact, 6, 6))
        E_matrix[:, 0, :] = 1
        E_matrix[:, 1, 0] = E_matrix[:, 2, 2] = self.miu_coef[0]
        E_matrix[:, 1, 1] = E_matrix[:, 2, 3] = -self.miu_coef[0]
        E_matrix[:, 3, 4] = self.miu_coef[1]
        E_matrix[:, 3, 5] = -self.miu_coef[1]

        return G_matrix, h_matrix, E_matrix

    def solve(
        self,
        pos,
        normal,
        gravity,
        gravity_center,
        retract_force=None,
        retract_weight=1.0,
    ):
        """
        Parameters
        -------------------
        pos: np.array [n, 3]. If n is different from self.num_contact, update self.num_contact=n and initialize again.
        normal: np.array [n, 3]. Direction is from the object to hand
        gravity: np.array [6]
        gravity_center: np.array [6]
        retract_force: np.array [n, 3]. The solved force part of contact_wrenches should be close to it.
        retract_weight: float.

        Returns
        -------------------
        contact_wrenches: np.array [n, 6]
        wrench_error: np.array [6]
        """
        # print(repr(pos), repr(normal), repr(gravity), repr(gravity_center))
        # print(repr(retract_force), repr(retract_weight))
        
        if pos.shape[0] != self.num_contact:
            self.num_contact = pos.shape[0]
            self.G_matrix, self.h_matrix, self.E_matrix = self.init_LCQP()

        axis_0, axis_1, axis_2 = utils_1axis_to_3axes(normal)
        # TODO: Do we need obb length to balance force and torque here?
        relative_pos = pos - gravity_center[None]
        grasp_matrix = np.zeros((self.num_contact, 6, 6))
        grasp_matrix[:, :3, 0] = grasp_matrix[:, 3:, 3] = axis_0
        grasp_matrix[:, :3, 1] = grasp_matrix[:, 3:, 4] = axis_1
        grasp_matrix[:, :3, 2] = grasp_matrix[:, 3:, 5] = axis_2
        grasp_matrix[:, 3:, 0] = np.cross(relative_pos, axis_0, axis=-1)
        grasp_matrix[:, 3:, 1] = np.cross(relative_pos, axis_1, axis=-1)
        grasp_matrix[:, 3:, 2] = np.cross(relative_pos, axis_2, axis=-1)

        param2force = grasp_matrix @ self.E_matrix  # [n, 6, 6]

        # [n, 6, 6] -> [6, n, 6] -> [6, 6n]
        flatten_param2force = np.transpose(param2force, (1, 0, 2)).reshape(6, -1)

        if retract_force is None:
            P_matrix = flatten_param2force.T @ flatten_param2force
            q_matrix = -gravity @ flatten_param2force
        else:
            semi_P_matrix = np.zeros((3 * self.num_contact + 6, 6 * self.num_contact))
            for i in range(self.num_contact):
                semi_P_matrix[3 * i : 3 * i + 3, 6 * i : 6 * i + 6] = (
                    retract_weight 
                ) * param2force[i, :3, :]
            semi_P_matrix[-6:, :] = flatten_param2force

            q_part = np.concatenate(
                [retract_weight * retract_force.reshape(-1), gravity]
            )

            P_matrix = semi_P_matrix.T @ semi_P_matrix
            q_matrix = -q_part @ semi_P_matrix

        # Minimize_x 1/2*x^T @ P_matrix @ x + q_matrix^T @ x
        # Subject to G_matrix @ x <= h_matrix
        solution = solve_qp(
            P_matrix, q_matrix, self.G_matrix, self.h_matrix, solver=self.solver_type
        )
        solution = solution.reshape(-1, 6)

        contact_wrenches = (param2force @ solution[..., None]).squeeze(
            axis=-1
        )  # [n, 6]
        wrench_error = np.sum(contact_wrenches, axis=0) - gravity

        return contact_wrenches, wrench_error,solution


def test_solve():
    # NOTE: Normal direction and contact wrenches direction: obj2hand
    num_contact = 2
    miu_coef = [0.5, 0.02]
    solver_type = "proxqp"  # Another choice: "clarabel"
    solver = ContactQP(num_contact, miu_coef, solver_type)

    # pos = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # normal = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    pos = np.array([[1, 0, 0], [-1, 0, 0]])
    normal = np.array([[1, 0, 0], [-1, 0, 0]])
    gravity = np.array([0, 0, -0.1, 0, 0, 0])
    gravity_center = np.array([0, 0, 0])

    contact_wrenches, wrench_error, solution = solver.solve(pos, normal, gravity, gravity_center)
    print("Contact Wrenches:\n", contact_wrenches[:, :3])
    print("Wrench Error:\n", wrench_error)
    print("Solution:\n", solution)

    assert contact_wrenches.shape == (num_contact, 6)
    assert wrench_error.shape == (6,)
    assert solution.shape == (num_contact, 6)


class QP_Solver:
    def __init__(self, 
                 num_affordance_contact=13, 
                 solver_type="proxqp",
                 hand_dim=16):
        self.solver_type = solver_type
        self.pre_retract_force = {}
        self.num_affordance_contact = num_affordance_contact
        self.hand_dim = hand_dim
        self.solved_num = 0
        
    def reset(self):
        self.solved_num = 0
    def solve_qp(self, contact_info_list, object_info_list):
        """
        NOTE: Normal direction and contact wrenches direction: obj2hand
        Input:
            contact_info_list: a list of dictionaries, each dictionary contains the following keys:
                "contact_points": np.array [n, 3]
                "normal": np.array [n, 3]
                "forces": np.array [n, 3]
                "env_id": int
                "num_contact": int
        
            object_info_list: a list of dictionaries, each dictionary contains the following keys:
                "object_weight":  float  Netwon
                "object_gravity_center": np.array [3,]  
                "miu_coef": list [miu, 0.02]
                "env_id": int
                
        output:
            target_normal_force_list: target normal force, a list of np.array [num_contact, 3] from hand 2 object 
            wrench_error_list: total wrench error for qp solutions, quantity of current grasp pose
        """
        
        num_env = len(contact_info_list)
        target_normal_force_list = []
        wrench_error_list = []
        self.solved_num += 1


        for i in range(num_env):
            contact_info = contact_info_list[i]
            object_info = object_info_list[i]
            num_contact = contact_info["num_contact"]
            if num_contact < 1:
                target_normal_force_list.append(np.zeros((1,3)))
                wrench_error_list.append(0)
                continue
            
            miu_coef = object_info["miu_coef"] 
            miu_coef[0] -= 0.2
            # f_scaling = 1.5  # NOTE: In real implementation, the leap hand max fingertip force is 1.5N  
            gravity = np.array([0,0.0,-0.1,0.0,0.0,0.0]).reshape(6) 
            gravity_center = object_info["object_gravity_center"]
        
            solver = ContactQP(
                num_contact=num_contact,
                miu_coef=miu_coef,
                solver_type=self.solver_type
            )
            
            retract_force=np.zeros((num_contact,3))
            for i, contact_id in enumerate(contact_info['contact_ids']):
                retract_force[i]=  np.zeros(3) if contact_id not in self.pre_retract_force.keys() else self.pre_retract_force[contact_id]
            
            contact_wrenches, wrench_error, solution = solver.solve(
                np.array(contact_info["points"]),
                np.array(contact_info["normals"]),
                gravity,
                gravity_center,
                retract_force,
                retract_weight=0.02 * (self.solved_num / 10 > 1)
                )
            
            target_force = contact_wrenches[:, :3] * object_info["object_weight"] * 10 

            target_normal_force = np.zeros_like(contact_wrenches[:, :3])
            for i in range(num_contact):
                target_normal_force[i] = np.abs(np.dot(
                    target_force[i], np.array(contact_info["normals"])[i]
                )) *  -np.array(contact_info["normals"])[i] 
            
            # print(wrench_error)
            target_normal_force_list.append(target_normal_force)
            wrench_error_list.append(np.max(np.abs(wrench_error[:3]))) # mearsuring quality of current grasp   NOTE: Only consider trans, without rot
            for i, contact_id in enumerate(contact_info['contact_ids']):
                self.pre_retract_force[contact_id] = deepcopy(contact_wrenches[i, :3])
        # print("target_normal_force_list:\n", target_normal_force_list,"\b",np.array( contact_info_list[0]['normal_forces']))
        # print(np.array(contact_info_list[0]['normal_forces']))
        # print(np.array(target_normal_force_list), "-----\n")
        
        return target_normal_force_list, wrench_error_list

    def qp_force_as_ref(self, contact_info_list, object_info_list):
        """
        NOTE: Normal direction and contact wrenches direction: obj2hand
        Input:
            contact_info_list: a list of dictionaries, each dictionary contains the following keys:
                "contact_points": np.array [n, 3]
                "normal": np.array [n, 3]
                "normal_forces" : np.array [n, 3]
                "forces": np.array [n, 3]
                "env_id": int
                "contact_ids": list of int
                "num_contact": int
        
            object_info_list: a list of dictionaries, each dictionary contains the following keys:
                "object_weight":  float  Netwon
                "object_gravity_center": np.array [3,]  
                "miu_coef": list [miu, 0.02]
                "env_id": int
        
        output: Directly input of the actor policy network, the force error of contact and desired from QP solver.
             forces_error: np.array [num_env, num_affordance_contact * 3]  affordance contact = 13 in RD grasp
        """
        target_force_list, wrench_error_list = self.solve_qp(contact_info_list, object_info_list)
        num_env = len(contact_info_list)
        forces_error = np.full((num_env, self.num_affordance_contact, 3), 0) # (32, 13, 3)
        for i in range(num_env):
            if contact_info_list[i]['num_contact'] < 1:
                wrench_error_list[i] = 0.1
                continue
            # if wrench_error_list[i] < 1e-3: # QP has solution
            for j, id in enumerate(contact_info_list[i]['contact_ids']):
                # forces_error[i, id] = np.clip(np.array(contact_info_list[i]["normal_forces"])[j] - target_force_list[i][j],
                #                               0, 10)
                forces_error[i, id] = np.array(contact_info_list[i]["normal_forces"])[j] - target_force_list[i][j]
                
            # else:
            #     wrench_error_list[i] = 0.1
        # print(np.array(wrench_error_list).reshape(-1))
        # print("\nContact normal forces:\n", np.array(contact_info_list[0]['normal_forces']))
        # print("\nTarge normal_forces: \n", target_force_list[0])
        # print("\nContact ID:\n", contact_info_list[0]['contact_ids'])
        # print("\nForce error:\n", forces_error[0])
        forces_error = forces_error.reshape(num_env, -1)
        wrench_error_list = np.array(wrench_error_list).reshape(num_env, -1)
        assert forces_error.shape == (num_env, self.num_affordance_contact * 3)
        assert len(wrench_error_list)== num_env
        
        
        return forces_error, wrench_error_list
    
    def qp_ik_torque_as_ref(self, contact_info_list, object_info_list):     
        """
        NOTE: Normal direction and contact wrenches direction: obj2hand
        Input:
            contact_info_list: a list of dictionaries, each dictionary contains the following keys:
                "contact_points": np.array [n, 3]
                "normal": np.array [n, 3]
                "normal_forces" : np.array [n, 3]
                "forces": np.array [n, 3]
                "env_id": int
                "contact_ids": list of int
                "num_contact": int
                "env_jacobians" : np.array [n, 3, 22] (Ur5-allegro)
        
            object_info_list: a list of dictionaries, each dictionary contains the following keys:
                "object_weight":  float  Netwon
                "object_gravity_center": np.array [3,]  
                "miu_coef": list [miu, 0.02]
                "env_id": int
        
        output: Directly input of the actor policy network, the torque error computed from target normal forces and 
        Contact normal forces , and QP's wrench error list.
             forces_error: np.array [num_env, 16]  
             wrench_error_list: np.array [num_env, 1]
        """
        target_force_list, wrench_error_list = self.solve_qp(contact_info_list, object_info_list)
        # print("target_force_list:\n", target_force_list[0])
        # print("contact_info_list:\n", np.array(contact_info_list[0]['normal_forces']),"\n")
        num_env = len(contact_info_list)
        forces_error = np.full((num_env, self.num_affordance_contact, 3), 0) # (32, 13, 3)
        tau_error_list = np.full((num_env, 16), 0, dtype=np.float64)

        # print(tau_error_list.shape)
        for i in range(num_env):
            if contact_info_list[i]['num_contact'] < 1 or wrench_error_list[i] > 1e-3:
                wrench_error_list[i] = 0.1
                continue
            # if wrench_error_list[i] < 1e-3: # QP has solution
            tau = np.zeros(22)
            for j in range(contact_info_list[i]['num_contact']):
                forces_error = np.array(contact_info_list[i]["normal_forces"])[j] - target_force_list[i][j]
                # print(forces_error.shape, contact_info_list[i]['env_jacobians'][j].shape)
                tau += np.dot(contact_info_list[i]['env_jacobians'][j].T, forces_error)
            tau_error_list[i] = deepcopy(tau[-16:])
            # print("Inner loop tau:\n", tau)
            # print("Inner loop tau_error_list:\n", tau_error_list[i]) 
        # print(tau_error_list.shape)
        # print("Outer loop:\n", tau_error_list)
        
        return tau_error_list, wrench_error_list
    
    def solve_inverse_dynamics(self, jacobian_list, target_force_list):
        """
        Input:
            jacobian_list: (n, 3, 22)
            target_force_list: (n, 3)  point from object to hand
        Output:
            desired_tau: (16,)
        """
        desired_tau = np.zeros(self.hand_dim)
        for target_f, jac in zip(target_force_list, jacobian_list):
            desired_tau += np.dot(jac[:, -self.hand_dim:].T, np.array(target_f).reshape(3, 1)).reshape(-1)
        return desired_tau
        
    def solve_qp_inverse_dynamics(self, contact_info_list, object_info_list):
        target_force_list, wrench_error_list = self.solve_qp(contact_info_list, object_info_list)
        return self.solve_inverse_dynamics(contact_info_list['jacobian_list'], target_force_list)
    
if __name__ == "__main__":
    # test_solve() # test qp solver
    
    #  Example usage
    solver = QP_Solver()
    # contact_info_list =  [{'env_index': 0, 'points': [[0.02092856783579544, -0.7442377491966132, 0.9465901056587425], [-0.00467646226136451, -0.744112384801279, 0.9229076551185225], [0.02591201626733104, -0.5725488382167216, 0.9461193526545346], [0.029870398757116647, -0.5689025186905432, 0.9362705670075544]], 'normals': [[0.42672597841289583, -0.34556921845349403, 0.8357552599924203], [-0.42982984037606076, -0.9025013170729567, 0.027160283574892485], [-0.1791356938437422, 0.5638328615824152, 0.8062275779151357], [-0.010591385236157732, 0.9996820210205419, -0.022884042629396886]], 'forces': [[0.29742126510442257, 1.4294455038648073, -9.574065050588871], [4.851746585920073, 12.898589486540153, -7.1876404087163035], [0.004958885371111085, -0.6396245612240173, -0.2974980388582418], [-1.6445896253272203, -11.495835662881634, -4.594561630574854]], 'contact_ids': [np.int32(2), np.int32(6), np.int32(11), np.int32(12)], 'num_contact': np.int32(4)}]
    # object_info_list =  [{'object_weight': np.array([2.943], dtype=np.float32), 'miu_coef': [np.float32(0.8), 0.02], 'object_gravity_center': np.array([ 0.01965238, -0.66543007,  0.88474435], dtype=np.float32), 'env_id': 0}]
    # solver.solve_qp(contact_info_list, object_info_list)


    # First set of data
    object_info_list_1 = [{
        'object_weight': np.array([2.943], dtype=np.float32),
        'miu_coef': [np.float32(0.8), 0.02],
        'object_gravity_center': np.array([0.04682863, -0.49414742, 0.9444819], dtype=np.float32),
        'env_id': 0
    }]

    contact_info_list_1 = [{
        'env_index': 0,
        'points': [
            [0.09507337910275, -0.5042775887108899, 0.9828495639231093],
            [0.03734005504008942, -0.5041071177783204, 0.9575619236472485],
            [-0.01931489834393183, -0.4825272134521277, 0.9653583193826412],
            [0.04194915626919651, -0.4553666285256208, 0.9978150496250722]
        ],
        'normals': [
            [0.2554727105997279, -0.26334952476364704, -0.9302584167561125],
            [0.025130569552794652, -0.9984612076507053, 0.049433503726192325],
            [-0.2608893449680761, -0.9233481735731669, -0.281717766641057],
            [-0.15926122034097676, 0.7214006075038558, 0.6739562502036544]
        ],
        'forces': [
            [2.1733123758914266, 1.9431252960465453, 4.670416619687225],
            [3.097552290069702, 7.10298231710517, 4.256510982164734],
            [-0.7380657606352252, 5.003338903835469, 5.968105380425194],
            [-4.468072417572358, -13.997719027508957, -11.825514424588508]
        ],
        'normal_forces': [
            [-1.0988375596640523, 1.132717261466603, 4.001221446022893],
            [-0.1709832383813151, 6.793325170111448, -0.3363354154740687],
            [1.5936664780654897, 5.640356956650007, 1.720898801084677],
            [2.7641778267081922, -12.520810522277003, -11.697354315104544]
        ],
        'contact_ids': [np.int32(3), np.int32(6), np.int32(9), np.int32(11)],
        'num_contact': np.int32(4)
    }
]
    # Second set of data
    object_info_list_2 = [{
        'object_weight': np.array([2.943], dtype=np.float32),
        'miu_coef': [np.float32(0.8), 0.02],
        'object_gravity_center': np.array([0.04776248, -0.49561864, 0.9453479], dtype=np.float32),
        'env_id': 0
    }]

    contact_info_list_2 = [{
        'env_index': 0,
        'points': [
            [0.09668017718836319, -0.5053487138431736, 0.9837635441232544],
            [0.037788653277015936, -0.505790813451351, 0.9604718428849899],
            [-0.014959733570628988, -0.4833839518338114, 0.9632020359584834],
            [0.04487701391767713, -0.4565845087686799, 0.9991168628506621]
        ],
        'normals': [
            [0.24859886719438154, -0.271536578980274, -0.9297669006290554],
            [0.06783861428162274, -0.9918932457235908, -0.10745097254222308],
            [-0.2306733219187912, -0.9430911099712451, -0.23951821819676805],
            [-0.18041487060694278, 0.7148478334378928, 0.6756056908382513]
        ],
        'forces': [
            [1.9854233149992355, 1.349743008572225, 4.198829540941103],
            [4.256432637844053, 7.074566671357694, 4.8756277142267574],
            [-0.6077784679399151, 5.2275662082385805, 5.985121197807132],
            [-5.676472598038551, -13.568266811949993, -12.084764094030898]
        ],
        'normal_forces': [
            [-0.9389239480188233, 1.0255565507799946, 3.5116025222804694],
            [-0.49198973691351827, 7.193562282743194, 0.7792726350908217],
            [1.4355766201414109, 5.869250665296122, 1.4906221113088272],
            [3.0381253432110937, -12.037795509876496, -11.376971113568187]
        ],
        'contact_ids': [np.int32(3), np.int32(6), np.int32(9), np.int32(11)],
        'num_contact': np.int32(4)
    }]

  
    # print(solver.qp_force_as_ref(contact_info_list_1, object_info_list_1))
    # print(solver.qp_force_as_ref(contact_info_list_2, object_info_list_2))