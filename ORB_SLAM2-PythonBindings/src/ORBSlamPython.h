#ifndef ORBSLAMPYTHON_H
#define ORBSLAMPYTHON_H

#include <memory>
#include <Python.h>
#include <boost/python.hpp>
#include <ORB_SLAM2/System.h>
#include <ORB_SLAM2/Tracking.h>
#include <exception>

class MappingOperationPython;


class ORBSlamPython
{
public:
    ORBSlamPython(std::string vocabFile, std::string settingsFile,
        ORB_SLAM2::System::eSensor sensorMode = ORB_SLAM2::System::eSensor::RGBD);
    ORBSlamPython(const char* vocabFile, const char* settingsFile,
        ORB_SLAM2::System::eSensor sensorMode = ORB_SLAM2::System::eSensor::RGBD);
    ~ORBSlamPython();
    
    bool initialize();
    bool isRunning();
    bool loadAndProcessMono(std::string imageFile, double timestamp);
    bool processMono(cv::Mat image, double timestamp);
    bool loadAndProcessStereo(std::string leftImageFile, std::string rightImageFile, double timestamp);
    bool processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp);
    bool loadAndProcessRGBD(std::string imageFile, std::string depthImageFile, double timestamp);
    bool processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp);
    void reset();
    void shutdown();
    ORB_SLAM2::Tracking::eTrackingState getTrackingState() const;
    unsigned int getNumFeatures() const;
    unsigned int getNumMatches() const;
    unsigned int getNumKeyframes() const;
    boost::python::list getAllKeyframesData() const;
    boost::python::list getAllMapPoints() const;
    // boost::python::tuple getInitialKeyframes() const;
    // boost::python::list getInitialKeyframemnIDs() const;
    boost::python::list getKeyframePoints() const;
    boost::python::list getTrajectoryPoints() const;
    // boost::python::tuple getAllMapPointsAndImagePoints() const;
    
    MappingOperationPython getAndPopMappingOperation() const;

    bool hasMappingOperation() const;
    uint getMappingOperationQueueSize() const;




    bool saveSettings(boost::python::dict settings) const;
    boost::python::dict loadSettings() const;
    void setMode(ORB_SLAM2::System::eSensor mode);
    void setRGBMode(bool rgb);
    void setUseViewer(bool useViewer);
    
    static bool saveSettingsFile(boost::python::dict settings, std::string settingsFilename);
    static boost::python::dict loadSettingsFile(std::string settingsFilename);
    
private:
    std::string vocabluaryFile;
    std::string settingsFile;
    ORB_SLAM2::System::eSensor sensorMode;
    std::shared_ptr<ORB_SLAM2::System> system;
    bool bUseViewer;
    bool bUseRGB;
};

class MappingOperationPython
{
public:
    MappingOperationPython(): meOperationType(ORB_SLAM2::MappingOperation::OprType::LocalMappingBA),
        m_opr(ORB_SLAM2::MappingOperation(ORB_SLAM2::MappingOperation::OprType::LocalMappingBA)){};
    MappingOperationPython(const ORB_SLAM2::MappingOperation& opr): m_opr(opr), meOperationType(opr.meOperationType){};

    boost::python::tuple getAssociatedKeyFrames() const;
    
    boost::python::tuple getAssociatedMapPointsAndImagePoints() const;
    
    ORB_SLAM2::MappingOperation::OprType meOperationType;

private:
    ORB_SLAM2::MappingOperation m_opr;
};


#endif // ORBSLAMPYTHON_H
