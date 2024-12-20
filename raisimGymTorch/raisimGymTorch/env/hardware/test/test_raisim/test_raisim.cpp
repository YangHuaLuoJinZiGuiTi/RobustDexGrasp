// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

// cpp library
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

#include <stack>


int main(int argc, char ** argv)
{

    raisim::World::setActivationKey("/home/wzj/.raisim/activation.raisim");
    raisim::RaiSimMsg::setFatalCallback([](){throw;});

    /// create raisim world
    raisim::World world;
    world.setTimeStep(0.001);
    world.setGravity(raisim::Vec<3>{0,0,0});

    /// create objects
    auto ground = world.addGround(0, "gnd");
    raisim::ArticulatedSystem* arm_hand_platform_ = world.addArticulatedSystem("/home/ubuntu/hand/github/vision_dex/rsc/ur5_allegro/ur5_allegro.urdf","",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(63)); 
    raisim::ArticulatedSystem* arctic = world.addArticulatedSystem("/home/ubuntu/hand/github/vision_dex/rsc/ycb_urdf_all/035_power_drill/035_power_drill.urdf","",{},raisim::COLLISION(1),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(63)); 

    const std::string contact_bodies[17] =  {"Allegro_base_link",
    "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip",
    "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip",
    "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip",
    "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"};
    
    for(int i = 0; i < 17 ;i++){
        std::string name = contact_bodies[i] + "/0";
        const std::string mat_name = arm_hand_platform_->getCollisionBody(name).getMaterial();
        std::cout << name << " material is " << mat_name << std::endl;
    }

    arctic->getCollisionBody("top/0").setMaterial("object");
    const std::string obj_mat_name = arctic->getCollisionBody("top/0").getMaterial();
    std::cout << " obj material is " << obj_mat_name << std::endl;

    double friction_list[8] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8};
    double random_friction = 0.01;//friction_list[std::rand() % 8];
    world.setMaterialPairProp("object", "object", random_friction+0.1, 0.0, 0.0);
    world.setMaterialPairProp("object", "finger", random_friction, 0.0, 0.0);
    world.setMaterialPairProp("finger", "finger", random_friction+0.1, 0.0, 0.0);

}
