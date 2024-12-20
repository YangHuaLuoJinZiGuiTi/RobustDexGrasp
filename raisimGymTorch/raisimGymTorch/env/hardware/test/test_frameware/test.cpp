#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

using namespace std;

// 抽象类：机械臂
class Arm {
public:
    virtual string getArmType() const = 0; // 纯虚函数，获取机械臂类型
    virtual ~Arm() = default;
};

// 抽象类：机械手
class Hand {
public:
    virtual string getHandType() const = 0; // 纯虚函数，获取机械手类型
    virtual ~Hand() = default;
};

// A型机械臂
class ArmA : public Arm {
public:
    string getArmType() const override {
        return "Arm A";
    }
};

// B型机械臂
class ArmB : public Arm {
public:
    string getArmType() const override {
        return "Arm B";
    }
};

// C型机械臂
class ArmC : public Arm {
public:
    string getArmType() const override {
        return "Arm C";
    }
};

// D型机械手
class HandD : public Hand {
public:
    string getHandType() const override {
        return "Hand D";
    }
};

// E型机械手
class HandE : public Hand {
public:
    string getHandType() const override {
        return "Hand E";
    }
};

// F型机械手
class HandF : public Hand {
public:
    string getHandType() const override {
        return "Hand F";
    }
};

// 硬件类
class Hardware {
private:
    unique_ptr<Arm> arm; // 使用智能指针管理机械臂
    unique_ptr<Hand> hand; // 使用智能指针管理机械手

    // 使用 unordered_map 存储机械臂和机械手的创建函数
    unordered_map<string, function<unique_ptr<Arm>()>> armFactory = {
        {"ArmA", []() { return make_unique<ArmA>(); }},
        {"ArmB", []() { return make_unique<ArmB>(); }},
        {"ArmC", []() { return make_unique<ArmC>(); }}
    };

    unordered_map<string, function<unique_ptr<Hand>()>> handFactory = {
        {"HandD", []() { return make_unique<HandD>(); }},
        {"HandE", []() { return make_unique<HandE>(); }},
        {"HandF", []() { return make_unique<HandF>(); }}
    };

public:
    template<typename T>
    void set_instance(const string& type, unordered_map<string, function<unique_ptr<T>()>>& factory, unique_ptr<T>& component) {
        auto it = factory.find(type);
        if (it != factory.end()) {
            component = factory[type](); //it->second; // 找到对应的函数
        } else {
            std::cout << "Key not found. use the default key: " << factory.begin()->first << std::endl;
            component = factory[factory.begin()->first](); //->second; // 返回第一个函数
        }
    }


    // 设置机械臂
    void setArm(const string& armType) {
        set_instance(armType, armFactory, arm);
    }

    // 设置机械手
    void setHand(const string& handType) {
        set_instance(handType, handFactory, hand);
    }

    // 获取机械臂和机械手的参数
    void getParameters() const {
        if (arm && hand) {
            cout << "Arm Type: " << arm->getArmType() << ", Hand Type: " << hand->getHandType() << endl;
        } else {
            cout << "Arm or Hand not set." << endl;
        }
    }
};

int main() {
    // 创建硬件对象
    Hardware hardware;
    
    // 设置机械臂和机械手
    hardware.setArm("ArmSSSA");
    hardware.setHand("HandD");
    
    // 获取并显示机械臂和机械手的参数
    hardware.getParameters();

    return 0;
}