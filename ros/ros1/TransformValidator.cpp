#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

class TransformValidator {
public:
    TransformValidator(const ros::NodeHandle& nh, const ros::NodeHandle& pnh) 
        : nh_(nh), pnh_(pnh) {
        
        // 订阅PX4位姿
        pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 10,
                        &TransformValidator::poseCallback, this);
        
        // 发布不同转换方法的结果
        method1_pub_ = pnh_.advertise<nav_msgs::Odometry>("method1_result", 10); // 旋转矩阵方法
        method2_pub_ = pnh_.advertise<nav_msgs::Odometry>("method2_result", 10); // 欧拉角方法
        method3_pub_ = pnh_.advertise<nav_msgs::Odometry>("method3_result", 10); // 四元数方法(MAVROS)
        
        // 发布路径可视化
        enu_path_pub_ = pnh_.advertise<nav_msgs::Path>("enu_path", 10);
        ned_path1_pub_ = pnh_.advertise<nav_msgs::Path>("ned_path1", 10);
        ned_path2_pub_ = pnh_.advertise<nav_msgs::Path>("ned_path2", 10);
        ned_path3_pub_ = pnh_.advertise<nav_msgs::Path>("ned_path3", 10);
        
        // 初始化路径
        enu_path_.header.frame_id = "map";
        ned_path1_.header.frame_id = "map";
        ned_path2_.header.frame_id = "map";
        ned_path3_.header.frame_id = "map";
        
        // 发布静态坐标转换
        publishStaticTransforms();
        
        ROS_INFO("坐标系转换验证节点已启动");
    }
    
    // 使用旋转矩阵方法进行坐标转换 (标准方法)
    geometry_msgs::Pose convertUsingRotationMatrix(const geometry_msgs::Pose& enu_pose) {
        geometry_msgs::Pose ned_pose;
        
        // 位置转换
        ned_pose.position.x = enu_pose.position.y;     // 北 = 东
        ned_pose.position.y = enu_pose.position.x;     // 东 = 北
        ned_pose.position.z = -enu_pose.position.z;    // 下 = -上
        
        // 创建ENU到NED的基本旋转矩阵
        Eigen::Matrix3d R_enu_ned;
        R_enu_ned << 0, 1, 0,    // NED.x = ENU.y
                     1, 0, 0,    // NED.y = ENU.x
                     0, 0, -1;   // NED.z = -ENU.z
        
        // 将ENU四元数转换为旋转矩阵
        Eigen::Quaterniond q_enu(enu_pose.orientation.w, 
                               enu_pose.orientation.x,
                               enu_pose.orientation.y,
                               enu_pose.orientation.z);
        Eigen::Matrix3d R_enu = q_enu.toRotationMatrix();
        
        // 应用转换: R_ned = R_enu_ned * R_enu * R_enu_ned.transpose()
        Eigen::Matrix3d R_ned = R_enu_ned * R_enu * R_enu_ned.transpose();
        
        // 转回四元数
        Eigen::Quaterniond q_ned(R_ned);
        q_ned.normalize();
        
        ned_pose.orientation.w = q_ned.w();
        ned_pose.orientation.x = q_ned.x();
        ned_pose.orientation.y = q_ned.y();
        ned_pose.orientation.z = q_ned.z();
        
        return ned_pose;
    }
    
    // 使用欧拉角方法进行坐标转换 (简单但有万向锁问题)
    geometry_msgs::Pose convertUsingEulerAngles(const geometry_msgs::Pose& enu_pose) {
        geometry_msgs::Pose ned_pose;
        
        // 位置转换
        ned_pose.position.x = enu_pose.position.y;     // 北 = 东
        ned_pose.position.y = enu_pose.position.x;     // 东 = 北
        ned_pose.position.z = -enu_pose.position.z;    // 下 = -上
        
        // 将四元数转换为欧拉角 (RPY)
        tf2::Quaternion q_enu;
        q_enu.setW(enu_pose.orientation.w);
        q_enu.setX(enu_pose.orientation.x);
        q_enu.setY(enu_pose.orientation.y);
        q_enu.setZ(enu_pose.orientation.z);
        
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_enu).getRPY(roll, pitch, yaw);
        
        // ENU到NED的欧拉角转换
        double ned_roll = pitch;
        double ned_pitch = roll;
        double ned_yaw = -yaw + M_PI/2;
        
        // 转回四元数
        tf2::Quaternion q_ned;
        q_ned.setRPY(ned_roll, ned_pitch, ned_yaw);
        q_ned.normalize();
        
        ned_pose.orientation.w = q_ned.w();
        ned_pose.orientation.x = q_ned.x();
        ned_pose.orientation.y = q_ned.y();
        ned_pose.orientation.z = q_ned.z();
        
        return ned_pose;
    }
    
    // 使用MAVROS四元数法 (标准实现)
    geometry_msgs::Pose convertUsingMavrosQuaternion(const geometry_msgs::Pose& enu_pose) {
        geometry_msgs::Pose ned_pose;
        
        // 位置转换
        ned_pose.position.x = enu_pose.position.y;     // 北 = 东
        ned_pose.position.y = enu_pose.position.x;     // 东 = 北
        ned_pose.position.z = -enu_pose.position.z;    // 下 = -上
        
        // 使用与MAVROS一致的转换方法
        tf2::Quaternion q_enu;
        q_enu.setW(enu_pose.orientation.w);
        q_enu.setX(enu_pose.orientation.x);
        q_enu.setY(enu_pose.orientation.y);
        q_enu.setZ(enu_pose.orientation.z);
        
        // 检查输入四元数是否归一化
        if (fabs(q_enu.length() - 1.0) > 1e-6) {
            q_enu.normalize();
        }
        
        // 创建ENU到NED的转换四元数：先绕Z轴旋转90度，再绕X轴旋转180度
        tf2::Quaternion q_enu_to_ned;
        q_enu_to_ned.setRPY(M_PI, 0.0, M_PI_2);
        
        // 应用转换: q_ned = q_enu_to_ned * q_enu
        tf2::Quaternion q_ned = q_enu_to_ned * q_enu;
        q_ned.normalize();
        
        ned_pose.orientation.w = q_ned.w();
        ned_pose.orientation.x = q_ned.x();
        ned_pose.orientation.y = q_ned.y();
        ned_pose.orientation.z = q_ned.z();
        
        return ned_pose;
    }
    
    // 比较两个姿态的差异
    void printPoseDiff(const std::string& label, const geometry_msgs::Pose& pose1, const geometry_msgs::Pose& pose2) {
        double pos_diff = sqrt(
            pow(pose1.position.x - pose2.position.x, 2) +
            pow(pose1.position.y - pose2.position.y, 2) +
            pow(pose1.position.z - pose2.position.z, 2)
        );
        
        // 计算四元数差异
        Eigen::Quaterniond q1(pose1.orientation.w, pose1.orientation.x, pose1.orientation.y, pose1.orientation.z);
        Eigen::Quaterniond q2(pose2.orientation.w, pose2.orientation.x, pose2.orientation.y, pose2.orientation.z);
        
        // 计算两个四元数之间的角度差
        double angle_diff = acos(std::min(1.0, std::abs(q1.dot(q2)))) * 2.0 * 180.0 / M_PI;
        
        ROS_INFO("%s - 位置差异: %.6f 米, 角度差异: %.6f 度", label.c_str(), pos_diff, angle_diff);
    }
    
    // 发布静态坐标系转换
    void publishStaticTransforms() {
        static tf2_ros::StaticTransformBroadcaster static_broadcaster;
        
        // 创建ENU到NED的转换
        geometry_msgs::TransformStamped enu_to_ned;
        enu_to_ned.header.stamp = ros::Time::now();
        enu_to_ned.header.frame_id = "map_enu";
        enu_to_ned.child_frame_id = "map_ned";
        
        // 设置平移部分（无平移）
        enu_to_ned.transform.translation.x = 0.0;
        enu_to_ned.transform.translation.y = 0.0;
        enu_to_ned.transform.translation.z = 0.0;
        
        // 设置旋转部分（先绕Z轴旋转90度，再绕X轴旋转180度）
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, M_PI_2);
        enu_to_ned.transform.rotation.x = q.x();
        enu_to_ned.transform.rotation.y = q.y();
        enu_to_ned.transform.rotation.z = q.z();
        enu_to_ned.transform.rotation.w = q.w();
        
        // 发布静态变换
        static_broadcaster.sendTransform(enu_to_ned);
    }
    
    // PX4位姿回调函数
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        // 获取当前ENU位姿
        geometry_msgs::Pose enu_pose = msg->pose;
        
        // 使用三种方法转换
        geometry_msgs::Pose ned_pose1 = convertUsingRotationMatrix(enu_pose);
        geometry_msgs::Pose ned_pose2 = convertUsingEulerAngles(enu_pose);
        geometry_msgs::Pose ned_pose3 = convertUsingMavrosQuaternion(enu_pose);
        
        // 比较结果
        printPoseDiff("方法1 vs 方法2 (矩阵vs欧拉角)", ned_pose1, ned_pose2);
        printPoseDiff("方法1 vs 方法3 (矩阵vs四元数)", ned_pose1, ned_pose3);
        printPoseDiff("方法2 vs 方法3 (欧拉角vs四元数)", ned_pose2, ned_pose3);
        
        // 创建Odometry消息用于发布
        nav_msgs::Odometry odom1, odom2, odom3;
        
        // 设置共同属性
        ros::Time current_time = ros::Time::now();
        
        // 方法1结果
        odom1.header.stamp = current_time;
        odom1.header.frame_id = "map";
        odom1.child_frame_id = "base_link";
        odom1.pose.pose = ned_pose1;
        
        // 方法2结果
        odom2.header.stamp = current_time;
        odom2.header.frame_id = "map";
        odom2.child_frame_id = "base_link";
        odom2.pose.pose = ned_pose2;
        
        // 方法3结果
        odom3.header.stamp = current_time;
        odom3.header.frame_id = "map";
        odom3.child_frame_id = "base_link";
        odom3.pose.pose = ned_pose3;
        
        // 发布结果
        method1_pub_.publish(odom1);
        method2_pub_.publish(odom2);
        method3_pub_.publish(odom3);
        
        // 更新路径
        updatePaths(msg->header.stamp, enu_pose, ned_pose1, ned_pose2, ned_pose3);
    }
    
    // 更新并发布路径
    void updatePaths(const ros::Time& time, const geometry_msgs::Pose& enu_pose, 
                    const geometry_msgs::Pose& ned_pose1, const geometry_msgs::Pose& ned_pose2, 
                    const geometry_msgs::Pose& ned_pose3) {
        // 创建路径点
        geometry_msgs::PoseStamped enu_pose_stamped;
        enu_pose_stamped.header.stamp = time;
        enu_pose_stamped.header.frame_id = "map";
        enu_pose_stamped.pose = enu_pose;
        
        geometry_msgs::PoseStamped ned_pose1_stamped = enu_pose_stamped;
        ned_pose1_stamped.pose = ned_pose1;
        
        geometry_msgs::PoseStamped ned_pose2_stamped = enu_pose_stamped;
        ned_pose2_stamped.pose = ned_pose2;
        
        geometry_msgs::PoseStamped ned_pose3_stamped = enu_pose_stamped;
        ned_pose3_stamped.pose = ned_pose3;
        
        // 添加到路径
        enu_path_.poses.push_back(enu_pose_stamped);
        ned_path1_.poses.push_back(ned_pose1_stamped);
        ned_path2_.poses.push_back(ned_pose2_stamped);
        ned_path3_.poses.push_back(ned_pose3_stamped);
        
        // 限制路径长度
        const size_t max_path_size = 1000;
        if (enu_path_.poses.size() > max_path_size) {
            enu_path_.poses.erase(enu_path_.poses.begin());
            ned_path1_.poses.erase(ned_path1_.poses.begin());
            ned_path2_.poses.erase(ned_path2_.poses.begin());
            ned_path3_.poses.erase(ned_path3_.poses.begin());
        }
        
        // 更新时间戳并发布
        enu_path_.header.stamp = time;
        ned_path1_.header.stamp = time;
        ned_path2_.header.stamp = time;
        ned_path3_.header.stamp = time;
        
        enu_path_pub_.publish(enu_path_);
        ned_path1_pub_.publish(ned_path1_);
        ned_path2_pub_.publish(ned_path2_);
        ned_path3_pub_.publish(ned_path3_);
    }
    
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // 订阅器
    ros::Subscriber pose_sub_;
    
    // 发布器
    ros::Publisher method1_pub_;
    ros::Publisher method2_pub_;
    ros::Publisher method3_pub_;
    ros::Publisher enu_path_pub_;
    ros::Publisher ned_path1_pub_;
    ros::Publisher ned_path2_pub_;
    ros::Publisher ned_path3_pub_;
    
    // 路径
    nav_msgs::Path enu_path_;
    nav_msgs::Path ned_path1_;
    nav_msgs::Path ned_path2_;
    nav_msgs::Path ned_path3_;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "transform_validator");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    TransformValidator validator(nh, pnh);
    
    ros::Rate rate(10); // 10Hz
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
} 