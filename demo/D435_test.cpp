#include "raisim/RaisimServer.hpp"
#include "raisim/server/Visuals.hpp"
#include "raisim/server/Charts.hpp"
#include "raisim/server/SerializationHelper.hpp"
#include "raisim/World.hpp"
#include "raisim/helper.hpp"
#include "raisim/object/ArticulatedSystem/JointAndBodies.hpp"
#include "raisim/sensors/Sensors.hpp"

//#define USE_UI

//#define LATEST_VERSION


bool needsSensorUpdate_without_ui(raisim::World *world, raisim::Sensor* rgb, raisim::Sensor* depth) {
  auto &objList = world->getObjList();

  for (auto *ob: objList) {
    if (ob->getObjectType() == raisim::ObjectType::ARTICULATED_SYSTEM) {
      auto as = dynamic_cast<raisim::ArticulatedSystem *>(ob);

#ifdef LATEST_VERSION
      for (auto &sensorSet: as->getSensorSets()) {
        for (auto &sensor : sensorSet->getSensors()) {
          sensor->lockMutex();
          if (sensor->getMeasurementSource() == raisim::Sensor::MeasurementSource::VISUALIZER &&
              sensor->getUpdateTimeStamp() + 1. / sensor->getUpdateRate()
                  < world->getWorldTime() + 1e-10) {
            sensor->unlockMutex();
            return true;
          }
          sensor->unlockMutex();
        }
      }

#else
    for (auto &sensor: as->getSensors()) {
      if (sensor.second->getMeasurementSource() == raisim::Sensor::MeasurementSource::VISUALIZER &&
          sensor.second->getUpdateTimeStamp() + 1. / sensor.second->getUpdateRate()
              < world->getWorldTime() + 1e-10) {
        sensor.second->setUpdateTimeStamp(world->getWorldTime());
        sensor.second->updatePose();
        return true;
      } 
    }
#endif

    }
  }


  return false;
}


void update_d435_without_ui(raisim::World *world, raisim::Sensor* rgb, raisim::Sensor* depth) {
    int width=640, height=480;
    int cnt = 0;
    //cv::Mat rgb_img = cv::Mat(height, width, CV_8UC1);
    //cv::Mat depth_img = cv::Mat(height, width, CV_8UC1);
    char *rgb_data, *depth_data;
    std::vector<float> rgbdata, depthdata;
    double sensorUpdateTime_ = world->getWorldTime();
    #ifdef LATEST_VERSION
    world->lockMutex();

    // update rgb
    rgb->lockMutex();
    #endif
    //rgb_data = raisim::server::get(rgb_data, &width, &height);
    auto &img = dynamic_cast<raisim::RGBCamera*>(rgb)->getImageBuffer();
    RSFATAL_IF(width * height * 4 != img.size(), "Image size mismatch. Sensor module not working properly")
    //rgb_data = raisim::server::getN(rgb_data, img.data(), width * height * 4);
    for (int i = 0; i < width * height * 4; i++) {
      if (img.data()[i] > 0) {
        cnt++;
        if (cnt > 100) {
          printf("rgb[%d]=%d\n", i , img.data()[i] );
          break;
        }
      }
    }
    #ifdef LATEST_VERSION
    rgb->setUpdateTimeStamp(sensorUpdateTime_);
    rgb->unlockMutex();


    // update depth
    depth->lockMutex();
    #endif
    //depth_data = raisim::server::get(depth_data, &width, &height);
    auto &depthArray = dynamic_cast<raisim::DepthCamera*>(depth)->getDepthArray();
    RSFATAL_IF(width * height != depthArray.size(), "depth size mismatch. Sensor module not working properly")
    //depth_data = raisim::server::getN(depth_data, depthArray.data(), width * height);
    for (int i = 0; i < width * height; i++) {
      if (depthArray.data()[i] > 0.00001) {
        cnt++;
        if (cnt > 100) {
          printf("depth[%d]=%f\n", i , depthArray.data()[i] );
          break;
        }
      }
    }
    #ifdef LATEST_VERSION
    depth->setUpdateTimeStamp(sensorUpdateTime_);
    rgb->unlockMutex();
    depth->unlockMutex();

    world->unlockMutex();
    #endif

}


int main(int argc, char **argv) {
  raisim::RaiSimMsg::setFatalCallback([](){throw;});

  raisim::World world;

  #ifdef USE_UI
  raisim::RaisimServer server(&world);
  #endif

  auto checkerBoard = world.addGround(0.0, "gnd");
  Eigen::VectorXd jointConfig(7);
  Eigen::VectorXd jointVel(6);

  jointConfig << 0., 0., 0., 1., 0., 0., 0.;
  jointVel << 0, 0, 0, 0, 0, 0;

  auto d435 = world.addArticulatedSystem("/home/ubuntu/hand/artigrasp/rsc/d435_camera/urdf/d435.urdf", "", {}, raisim::COLLISION(0), raisim::COLLISION(-1)); // only collision itself
  d435->setName("d435");
  d435->setGeneralizedForce(Eigen::VectorXd::Zero(d435->getDOF()));
  d435->setState(jointConfig, jointVel);
  d435->setGeneralizedCoordinate(jointConfig);
  d435->setGeneralizedVelocity(jointVel);
  

  #ifdef LATEST_VERSION
  auto depthSensor1 = d435->getSensorSet("d435link")->getSensor<raisim::DepthCamera>("depth");
  #else
  auto depthSensor1 = d435->getSensor<raisim::DepthCamera>("d435link:depth");
  #endif
  depthSensor1->setMeasurementSource(raisim::Sensor::MeasurementSource::VISUALIZER);
//  depthSensor1->setMeasurementSource(raisim::Sensor::MeasurementSource::RAISIM); // uncomment this line if you want to update the sensor using Raisim (CPU)

  #ifdef LATEST_VERSION
  auto rgbCamera1 = d435->getSensorSet("d435link")->getSensor<raisim::RGBCamera>("color");
  #else
  auto rgbCamera1 = d435->getSensor<raisim::RGBCamera>("d435link:color");
  #endif
  rgbCamera1->setMeasurementSource(raisim::Sensor::MeasurementSource::VISUALIZER);

  /// launch raisim server raisim_UI ****************
  #ifdef USE_UI
    server.launchServer();
  #endif

  auto* ball = world.addSphere(0.1, 1.0);
  ball->setPosition(0, 0, 0.2);
  ball->setAppearance("red");
  int flag = 0;
  while (1) {
    RS_TIMED_LOOP(int(world.getTimeStep()*1e6))
    flag++;
    if (flag < 100) {
      ball->setVelocity(0, 5, 0, 0, 0, 0);
    } else if (flag < 200) {
      ball->setVelocity(5, 0, 0, 0, 0, 0);
    } else if (flag < 300) {
      ball->setVelocity(0, -5, 0, 0, 0, 0);
    } else if (flag < 400) {
      ball->setVelocity(-5, 0, 0, 0, 0, 0);
    } else {
      flag = 0;
    }

    #ifdef USE_UI
    server.integrateWorldThreadSafe();
    #else

    if (needsSensorUpdate_without_ui(&world, rgbCamera1, depthSensor1)) {
      update_d435_without_ui(&world, rgbCamera1, depthSensor1);
    }
    world.integrate();
    #endif
  }

  #ifdef USE_UI
  server.killServer();
  #endif
  return 0;
}