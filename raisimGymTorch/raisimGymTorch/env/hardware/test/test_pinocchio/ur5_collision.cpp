#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/collision/collision.hpp"

#include <iostream>

int main(int /*argc*/, char ** /*argv*/)
{
  using namespace pinocchio;
  const std::string robots_model_path = "/home/ubuntu/hand/UR5_IK/rsc";

  // You should change here to set up your own URDF file
  const std::string urdf_filename = robots_model_path + std::string("/ur5.urdf");
  // You should change here to set up your own SRDF file
  const std::string srdf_filename = robots_model_path + std::string("/ur5.srdf");

  // Load the URDF model contained in urdf_filename
  Model model;
  pinocchio::urdf::buildModel(urdf_filename, model);

  // Build the data associated to the model
  Data data(model);

  // Load the geometries associated to model which are contained in the URDF file
  GeometryModel geom_model;
  pinocchio::urdf::buildGeom(
    model, urdf_filename, pinocchio::COLLISION, geom_model, robots_model_path);

  // Add all possible collision pairs and remove the ones collected in the SRDF file
  geom_model.addAllCollisionPairs();
  pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);

  // Build the data associated to the geom_model
  GeometryData geom_data(geom_model); // contained the intermediate computations, like the placement
                                      // of all the geometries with respect to the world frame

  // Load the reference configuration of the robots (this one should be collision free)
  pinocchio::srdf::loadReferenceConfigurations(model, srdf_filename); 

  // load the joint set
  const Model::ConfigVectorType & q = model.referenceConfigurations["collision_check"];
  // test all the collision pairs : 124us for all joints
  // and 65us for stopping immediately if it find collision at the first time. (set true in the last parameter)
    auto starttime = std::chrono::system_clock::now();
    //for (int i = 0; i < 1000; i++)
    bool ret = computeCollisions(model, data, geom_model, geom_data, q, true); 
    auto diff_time = std::chrono::system_clock::now() - starttime;
    std::cout << "collision cost time: " <<diff_time.count() / 1e9 << std::endl;

    if (ret == true) {
        std::cout << "have collision " << std::endl;
    } else {
        std::cout << "donot have collision " << std::endl;
    }

  // Print the status of all the collision pairs
  for (size_t k = 0; k < geom_model.collisionPairs.size(); ++k)
  {
    const CollisionPair & cp = geom_model.collisionPairs[k];
    const hpp::fcl::CollisionResult & cr = geom_data.collisionResults[k];

    std::cout << "collision pair: " << cp.first << " , " << cp.second << " - collision: ";
    std::cout << (cr.isCollision() ? "yes" : "no") << std::endl;
  }

  return 0;
}
