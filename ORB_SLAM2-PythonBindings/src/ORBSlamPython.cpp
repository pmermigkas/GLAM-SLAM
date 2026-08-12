#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <opencv2/core/core.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <ORB_SLAM2/KeyFrame.h>
#include <ORB_SLAM2/Converter.h>
#include <ORB_SLAM2/Tracking.h>
#include <ORB_SLAM2/MapPoint.h>
#include "ORBSlamPython.h"



#if (PY_VERSION_HEX >= 0x03000000)
static void* init_ar() {
#else
static void init_ar() {
#endif
    Py_Initialize();

    import_array();
    return NULL;
}

class ScopedGILRelease {
public:
    inline ScopedGILRelease() { m_thread_state = PyEval_SaveThread(); }
    inline ~ScopedGILRelease() { PyEval_RestoreThread(m_thread_state); }
private:
    PyThreadState * m_thread_state;
};

void translate_exception(const std::exception& e) {
    // Use the Python C API to set the exception
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MODULE(orbslam2)
{
    init_ar();

    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::enum_<ORB_SLAM2::Tracking::eTrackingState>("TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM2::Tracking::eTrackingState::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM2::Tracking::eTrackingState::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM2::Tracking::eTrackingState::NOT_INITIALIZED)
        .value("OK", ORB_SLAM2::Tracking::eTrackingState::OK)
        .value("LOST", ORB_SLAM2::Tracking::eTrackingState::LOST);
    
    boost::python::enum_<ORB_SLAM2::System::eSensor>("Sensor")
        .value("MONOCULAR", ORB_SLAM2::System::eSensor::MONOCULAR)
        .value("STEREO", ORB_SLAM2::System::eSensor::STEREO)
        .value("RGBD", ORB_SLAM2::System::eSensor::RGBD);

    boost::python::class_<ORBSlamPython, boost::noncopyable>("System", boost::python::init<const char*, const char*, boost::python::optional<ORB_SLAM2::System::eSensor>>())
        .def(boost::python::init<std::string, std::string, boost::python::optional<ORB_SLAM2::System::eSensor>>())
        .def("initialize", &ORBSlamPython::initialize)
        .def("load_and_process_mono", &ORBSlamPython::loadAndProcessMono)
        .def("process_image_mono", &ORBSlamPython::processMono)
        .def("load_and_process_stereo", &ORBSlamPython::loadAndProcessStereo)
        .def("process_image_stereo", &ORBSlamPython::processStereo)
        .def("load_and_process_rgbd", &ORBSlamPython::loadAndProcessRGBD)
        .def("process_image_rgbd", &ORBSlamPython::processRGBD)
        .def("shutdown", &ORBSlamPython::shutdown)
        .def("is_running", &ORBSlamPython::isRunning)
        .def("reset", &ORBSlamPython::reset)
        .def("set_mode", &ORBSlamPython::setMode)
        .def("set_use_viewer", &ORBSlamPython::setUseViewer)
        .def("get_num_keyframes", &ORBSlamPython::getNumKeyframes)
        .def("get_all_keyframes_data", &ORBSlamPython::getAllKeyframesData)
        .def("get_keyframe_points", &ORBSlamPython::getKeyframePoints)
        .def("get_trajectory_points", &ORBSlamPython::getTrajectoryPoints)

        .def("get_and_pop_mapping_operation", &ORBSlamPython::getAndPopMappingOperation)
        .def("has_mapping_operation", &ORBSlamPython::hasMappingOperation)
        .def("get_mapping_operation_queue_size", &ORBSlamPython::getMappingOperationQueueSize)
    
        
        
        .def("get_tracking_state", &ORBSlamPython::getTrackingState)
        .def("get_num_features", &ORBSlamPython::getNumFeatures)
        .def("get_num_matched_features", &ORBSlamPython::getNumMatches)
        .def("save_settings", &ORBSlamPython::saveSettings)
        .def("load_settings", &ORBSlamPython::loadSettings)
        .def("save_settings_file", &ORBSlamPython::saveSettingsFile)
        .staticmethod("save_settings_file")
        .def("load_settings_file", &ORBSlamPython::loadSettingsFile)
        .staticmethod("load_settings_file");



    // Expose the OprType enum
    boost::python::enum_<ORB_SLAM2::MappingOperation::OprType>("OprType")
        .value("LocalMappingBA",ORB_SLAM2::MappingOperation::LocalMappingBA)
        .value("LoopClosingBA", ORB_SLAM2::MappingOperation::LoopClosingBA);

    // Expose the MappingOperation class
    boost::python::class_<MappingOperationPython>("MappingOperation", boost::python::init<ORB_SLAM2::MappingOperation::OprType>())
        .def("get_associated_keyframes", &MappingOperationPython::getAssociatedKeyFrames)
        .def("get_associated_mappoints_and_imagepoints", &MappingOperationPython::getAssociatedMapPointsAndImagePoints)
        .def_readwrite("meOperationType", &MappingOperationPython::meOperationType);

}

ORBSlamPython::ORBSlamPython(std::string vocabFile, std::string settingsFile, ORB_SLAM2::System::eSensor sensorMode)
    : vocabluaryFile(vocabFile),
    settingsFile(settingsFile),
    sensorMode(sensorMode),
    system(nullptr),
    bUseViewer(false),
    bUseRGB(true)
{
    
}

ORBSlamPython::ORBSlamPython(const char* vocabFile, const char* settingsFile, ORB_SLAM2::System::eSensor sensorMode)
    : vocabluaryFile(vocabFile),
    settingsFile(settingsFile),
    sensorMode(sensorMode),
    system(nullptr),
    bUseViewer(false),
    bUseRGB(true)
{

}

ORBSlamPython::~ORBSlamPython()
{
}

bool ORBSlamPython::initialize()
{
    system = std::make_shared<ORB_SLAM2::System>(vocabluaryFile, settingsFile, sensorMode, bUseViewer);
    return true;
}

bool ORBSlamPython::isRunning()
{
    return system != nullptr;
}

void ORBSlamPython::reset()
{
    if (system)
    {
        system->Reset();
    }
}

bool ORBSlamPython::loadAndProcessMono(std::string imageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat im = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (bUseRGB)
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    }
    return this->processMono(im, timestamp);
}

bool ORBSlamPython::processMono(cv::Mat image, double timestamp)
{
    if (!system)
    {
        return false;
    }

    // This releases the Python GIL for the duration of this function
    ScopedGILRelease gil_release;

    if (image.data)
    {
        cv::Mat pose = system->TrackMonocular(image, timestamp);
        return !pose.empty();
    }
    else
    {
        return false;
    }
}

bool ORBSlamPython::loadAndProcessStereo(std::string leftImageFile, std::string rightImageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat leftImage = cv::imread(leftImageFile, cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightImageFile, cv::IMREAD_COLOR);
    if (bUseRGB) {
        cv::cvtColor(leftImage, leftImage, cv::COLOR_BGR2RGB);
        cv::cvtColor(rightImage, rightImage, cv::COLOR_BGR2RGB);
    }
    return this->processStereo(leftImage, rightImage, timestamp);
}

bool ORBSlamPython::processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp)
{
    if (!system)
    {
        return false;
    }
    if (leftImage.data && rightImage.data) {
        cv::Mat pose = system->TrackStereo(leftImage, rightImage, timestamp);
        return !pose.empty();
    }
    else
    {
        return false;
    }
}

bool ORBSlamPython::loadAndProcessRGBD(std::string imageFile, std::string depthImageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat im = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (bUseRGB)
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    }
    cv::Mat imDepth = cv::imread(depthImageFile, cv::IMREAD_UNCHANGED);
    return this->processRGBD(im, imDepth, timestamp);
}

bool ORBSlamPython::processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp)
{
    if (!system)
    {
        return false;
    }
    if (image.data && depthImage.data)
    {
        cv::Mat pose = system->TrackRGBD(image, depthImage, timestamp);
        return !pose.empty();
    }
    else
    {
        return false;
    }
}

void ORBSlamPython::shutdown()
{
    if (system)
    {
        system->Shutdown();
        system.reset();
    }
}

ORB_SLAM2::Tracking::eTrackingState ORBSlamPython::getTrackingState() const
{
    if (system)
    {
        return static_cast<ORB_SLAM2::Tracking::eTrackingState>(system->GetTrackingState());
    }
    return ORB_SLAM2::Tracking::eTrackingState::SYSTEM_NOT_READY;
}

unsigned int ORBSlamPython::getNumFeatures() const
{
    if (system)
    {
        return system->GetTracker()->mCurrentFrame.mvKeys.size();
    }
    return 0;
}

unsigned int ORBSlamPython::getNumMatches() const
{
    if (system)
    {
        // This code is based on the display code in FrameDrawer.cc, with a little extra safety logic to check the length of the vectors.
        ORB_SLAM2::Tracking* pTracker = system->GetTracker();
        unsigned int matches = 0;
        unsigned int num = pTracker->mCurrentFrame.mvKeys.size();
        if (pTracker->mCurrentFrame.mvpMapPoints.size() < num)
        {
            num = pTracker->mCurrentFrame.mvpMapPoints.size();
        }
        if (pTracker->mCurrentFrame.mvbOutlier.size() < num)
        {
            num = pTracker->mCurrentFrame.mvbOutlier.size();
        }
        for(unsigned int i = 0; i < num; ++i)
        {
            ORB_SLAM2::MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP && !pTracker->mCurrentFrame.mvbOutlier[i] && pMP->Observations() > 0)
            {
                ++matches;
            }
        }
        return matches;
    }
    return 0;
}


unsigned int ORBSlamPython::getNumKeyframes() const
{
    if (!system)
    {
        return 0;
    }

    vector<ORB_SLAM2::KeyFrame*> vpKFs = system->GetKeyFrames();
    unsigned int numKeyframes = vpKFs.size();

    return numKeyframes;

}



boost::python::list ORBSlamPython::getAllKeyframesData() const
{
    // If the SLAM system isn't initialized, return an empty list immediately.
    if (!system)
    {
        return boost::python::list();
    }
    
    // This is the main list that will be returned to Python.
    boost::python::list all_keyframes_data;
    
    


    
    // Get associated 3D MapPoints and their corresponding 2D ImagePoints
    // Get pointer to map
    ORB_SLAM2::Map* map = system->GetMap();
    // Get MapPoints
    std::vector<ORB_SLAM2::MapPoint*> Mps = map->GetAllMapPoints();
    // std::cout << "MapPoints size is: " << Mps.size() << std::endl;
    
    boost::python::list mappoints_list;
    boost::python::list imagepoints_list;
    std::map<int, boost::python::list> mappoints_dict;
    std::map<int, boost::python::list> imagepoints_dict;
    for (size_t i = 0; i < Mps.size(); ++i)
    {
        ORB_SLAM2::MapPoint* pMP = Mps[i];
        
        // Only include points that are valid (not null and not marked as bad).
        if (pMP && !pMP->isBad())
        {
            // Get the 3D world coordinates of the MapPoint.
            cv::Mat wp = pMP->GetWorldPos();
            
            // Get the index of the map point in the reference keyframe (the first it was observed)
            ORB_SLAM2::KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            int pReKFId = pRefKF->mnId;
            int imagepoint_idx = pMP->GetIndexInKeyFrame(pRefKF);
            
            //Find the 2D coordinates in the ref keyframe of the mappoint
            // TODO: There is also mvKeysUn (undistorted)
            cv::KeyPoint keypoint = pRefKF->mvKeys[imagepoint_idx];

            // Create a tuple for the 3D point coordinates and 2D image coordinates 
            boost::python::tuple mappoint = boost::python::make_tuple(wp.at<float>(0,0), wp.at<float>(1,0), wp.at<float>(2,0));
            boost::python::tuple imagepoint = boost::python::make_tuple(int(keypoint.pt.x), int(keypoint.pt.y));
            
            // Put mappoints in dict with key their kfId
            mappoints_dict[pReKFId].append(mappoint);
            imagepoints_dict[pReKFId].append(imagepoint);
        }
    }        


    // Get all keyframes currently in the map from the SLAM system.
    std::vector<ORB_SLAM2::KeyFrame*> vpKFs = system->GetKeyFrames();

    std::vector<ORB_SLAM2::KeyFrame*> vpKFs_sorted;
    vpKFs_sorted.resize(vpKFs.size());
    std::copy(vpKFs.begin(), vpKFs.end(), vpKFs_sorted.begin());

    // Sort the vector of Keyframes we acquired so that we can index it
    std::sort(vpKFs_sorted.begin(), vpKFs_sorted.end(), [](const ORB_SLAM2::KeyFrame* a, const ORB_SLAM2::KeyFrame* b) {
        return a->mnId < b->mnId;
    });

    for (const auto& pair : mappoints_dict) 
    {
        int key = pair.first;
        ORB_SLAM2::KeyFrame* kf = vpKFs_sorted[key];

        // This is where the conversion from cv::Mat to boost::python::list happens.
        // It relies on a pre-registered converter.
        cv::Mat pose = kf->GetPose();
        boost::python::list pose_as_list;
        for (int i = 0; i < pose.rows; ++i) {
            const float* data = (const float*)pose.ptr(i);
            for (int j = 0; j < pose.cols; ++j) {
                pose_as_list.append(data[j]);
            }
        }

        // Assemble the final tuple for this keyframe with all the collected data.
        boost::python::tuple keyframe_tuple = boost::python::make_tuple(
            kf->mnFrameId,
            kf->mnId,
            pose_as_list,
            mappoints_dict[key],
            imagepoints_dict[key]
        );

        // Add the tuple for this keyframe to the main list.
        all_keyframes_data.append(keyframe_tuple);
    }
    

    // Return the complete list of keyframe tuples.
    return all_keyframes_data;
}

boost::python::list ORBSlamPython::getKeyframePoints() const
{
    if (!system)
    {
        return boost::python::list();
    }

    // This is copied from the ORB_SLAM2 System.SaveKeyFrameTrajectoryTUM function, with some changes to output a python tuple.
    vector<ORB_SLAM2::KeyFrame*> vpKFs = system->GetKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM2::KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    boost::python::list trajectory;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM2::KeyFrame* pKF = vpKFs[i];

        // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        cv::Mat t = pKF->GetCameraCenter();
        trajectory.append(boost::python::make_tuple(
                              pKF->mTimeStamp,
                              R.at<float>(0,0),
                              R.at<float>(0,1),
                              R.at<float>(0,2),
                              t.at<float>(0),
                              R.at<float>(1,0),
                              R.at<float>(1,1),
                              R.at<float>(1,2),
                              t.at<float>(1),
                              R.at<float>(2,0),
                              R.at<float>(2,1),
                              R.at<float>(2,2),
                              t.at<float>(2)
                              ));
    }

    return trajectory;
}

MappingOperationPython ORBSlamPython::getAndPopMappingOperation() const
{
    if (!system)
    {
        // Return a default-constructed Python object if the system is not ready
        return MappingOperationPython();
    }
    // Get pointer to map
    ORB_SLAM2::Map* map = system->GetMap();
    ORB_SLAM2::MappingOperation opr = map->getAndPopMappingOperation();
    
    // Directly construct and return the Python wrapper. Boost.Python handles the rest.
    return MappingOperationPython(opr);
}

bool ORBSlamPython::hasMappingOperation() const
{
    if (!system)
    {
        return false;
    }

    ORB_SLAM2::Map* map = system->GetMap();
    return map->hasMappingOperation();
}

uint ORBSlamPython::getMappingOperationQueueSize() const
{
    if (!system)
    {
        return 0;
    }

    ORB_SLAM2::Map* map = system->GetMap();
    return map->getMappingOperationQueueSize();
}

boost::python::list ORBSlamPython::getTrajectoryPoints() const
{
    if (!system)
    {
        return boost::python::list();
    }

    // This is copied from the ORB_SLAM2 System.SaveTrajectoryKITTI function, with some changes to output a python tuple.
    vector<ORB_SLAM2::KeyFrame*> vpKFs = system->GetKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM2::KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    // Of course, if we have no keyframes, then just use the identity matrix.
    cv::Mat Two = cv::Mat::eye(4,4,CV_32F);
    if (vpKFs.size() > 0) {
        cv::Mat Two = vpKFs[0]->GetPoseInverse();
    }

    boost::python::list trajectory;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.



    
    // // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    // std::list<ORB_SLAM2::KeyFrame*>::iterator lRit = system->GetTracker()->tracked_frames.begin().reference_keyframe;
    // std::list<double>::iterator lT = system->GetTracker()->mlFrameTimes.begin();
    // for(std::list<cv::Mat>::iterator lit=system->GetTracker()->tracked_frames.begin().relative_frame_pose, lend=system->GetTracker()->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    for(std::list<ORB_SLAM2::Tracking::TrackedFrame>::iterator lit=system->GetTracker()->tracked_frames.begin(), lend=system->GetTracker()->tracked_frames.end() ; lit!=lend ; lit++)
    {
        ORB_SLAM2::KeyFrame* pKF = (*lit).reference_keyframe;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF != NULL && pKF->isBad())
        {
            ORB_SLAM2::KeyFrame* pKFParent;

            // std::cout << "bad parent" << std::endl;
            Trw = Trw*pKF->mTcp;
            pKFParent = pKF->GetParent();
            if (pKFParent == pKF) {
                // We've found a frame that is it's own parent, presumably a root or something. Break out
                break;
            } else {
                pKF = pKFParent;
            }
        }
        if (pKF != NULL && !pKF->isBad()) {
            Trw = Trw*pKF->GetPose()*Two;

            cv::Mat Tcw = ((*lit).relative_frame_pose)*Trw;
            cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
            cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

            trajectory.append(boost::python::make_tuple(
                                (*lit).time,
                                Rwc.at<float>(0,0),
                                Rwc.at<float>(0,1),
                                Rwc.at<float>(0,2),
                                twc.at<float>(0),
                                Rwc.at<float>(1,0),
                                Rwc.at<float>(1,1),
                                Rwc.at<float>(1,2),
                                twc.at<float>(1),
                                Rwc.at<float>(2,0),
                                Rwc.at<float>(2,1),
                                Rwc.at<float>(2,2),
                                twc.at<float>(2)
                            ));
        }
    }

    return trajectory;
}

void ORBSlamPython::setMode(ORB_SLAM2::System::eSensor mode)
{
    sensorMode = mode;
}

void ORBSlamPython::setUseViewer(bool useViewer)
{
    bUseViewer = useViewer;
}

void ORBSlamPython::setRGBMode(bool rgb)
{
    bUseRGB = rgb;
}

bool ORBSlamPython::saveSettings(boost::python::dict settings) const
{
    return ORBSlamPython::saveSettingsFile(settings, settingsFile);
}

boost::python::dict ORBSlamPython::loadSettings() const
{
    return ORBSlamPython::loadSettingsFile(settingsFile);
}

bool ORBSlamPython::saveSettingsFile(boost::python::dict settings, std::string settingsFilename)
{
    cv::FileStorage fs(settingsFilename.c_str(), cv::FileStorage::WRITE);
    
    boost::python::list keys = settings.keys();
    for (int index = 0; index < boost::python::len(keys); ++index)
    {
        boost::python::extract<std::string> extractedKey(keys[index]);
        if (!extractedKey.check())
        {
            continue;
        }
        std::string key = extractedKey;
        
        boost::python::extract<int> intValue(settings[key]);
        if (intValue.check())
        {
            fs << key << int(intValue);
            continue;
        }
        
        boost::python::extract<float> floatValue(settings[key]);
        if (floatValue.check())
        {
            fs << key << float(floatValue);
            continue;
        }
        
        boost::python::extract<std::string> stringValue(settings[key]);
        if (stringValue.check())
        {
            fs << key << std::string(stringValue);
            continue;
        }
    }
    
    return true;
}






boost::python::tuple MappingOperationPython::getAssociatedKeyFrames() const
{

    if (!system)
    {
        // Return a tuple of empty lists if the system is not ready
        return boost::python::make_tuple(
            boost::python::list(),
            boost::python::list(),
            boost::python::list()
        );
    }
    
    boost::python::list keyframe_mnIDs;
    boost::python::list keyframe_kfIds;
    boost::python::list keyframe_poses;
    
    std::vector<ORB_SLAM2::KeyFrame*> vpKFs = m_opr.getAssociatedKeyFrames();
    for (ORB_SLAM2::KeyFrame* kf : vpKFs) {
        keyframe_mnIDs.append(kf->mnFrameId);
        keyframe_kfIds.append(kf->mnId);
        // This is where the conversion from cv::Mat to boost::python::list happens.
        // It relies on a pre-registered converter.
        
        cv::Mat pose = kf->GetPose();
        boost::python::list pose_as_list;
        for (int i = 0; i < pose.rows; ++i) {
            const float* data = (const float*)pose.ptr(i);
            for (int j = 0; j < pose.cols; ++j) {
                pose_as_list.append(data[j]);
            }
        }
        keyframe_poses.append(pose_as_list);

        // std::cout << "Appended (in wrapper) a Keyframe with mnFrameId=" << kf->mnFrameId << " and mnId=" << kf->mnId << "." << std::endl;
    }

    // Return all three lists as a single Boost.Python tuple
    return boost::python::make_tuple(keyframe_mnIDs, keyframe_kfIds, keyframe_poses);
}



boost::python::tuple MappingOperationPython::getAssociatedMapPointsAndImagePoints() const
{
    if (!system)
    {
        return boost::python::tuple();
    }
    // Get MapPoints
    std::vector<ORB_SLAM2::MapPoint*> Mps = m_opr.getAssociatedMapPoints();
    // std::cout << "MapPoints size is: " << Mps.size() << std::endl;
    
    boost::python::list mappoints, imagepoints, kfIds;
    boost::python::tuple mappoints_and_imagepoints;
    for(size_t i=0; i<Mps.size(); i++)    {
        ORB_SLAM2::MapPoint* pMP = Mps[i];
        
        if (pMP != NULL && !pMP->isBad())
        {
            cv::Mat wp = pMP->GetWorldPos();
            
            // std::map<KeyFrame*,size_t> mObservations = pPM->GetObservations();


            // Get the index of the map point in the reference keyframe (the first it was observed)
            ORB_SLAM2::KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            int pReKFId = pRefKF->mnId;
            int imagepoint_idx = pMP->GetIndexInKeyFrame(pRefKF);
            
            //Find the 2D coordinates in the ref keyframe of the mappoint
            // TODO: There is also mvKeysUn (undistorted)
            cv::KeyPoint keypoint = pRefKF->mvKeys[imagepoint_idx];
            
            // Create a tuple for the 3D point coordinates and 2D image coordinates 
            boost::python::tuple mappoint = boost::python::make_tuple(wp.at<float>(0,0), wp.at<float>(1,0), wp.at<float>(2,0));
            boost::python::tuple imagepoint = boost::python::make_tuple(int(keypoint.pt.x), int(keypoint.pt.y));


            mappoints.append(mappoint);
            imagepoints.append(imagepoint);
            kfIds.append(pReKFId);
        }
    }
    // Append a new tuple containing both the 3D point data and 2D point data
    mappoints_and_imagepoints = boost::python::make_tuple(mappoints, imagepoints, kfIds);
    // std::cout << "Finished a vector of MapPoints and ImagePoints of size " << boost::python::len(mappoints) << std::endl;
    return mappoints_and_imagepoints;
}

























// Helpers for reading cv::FileNode objects into python objects.
boost::python::list readSequence(cv::FileNode fn, int depth=10);
boost::python::dict readMap(cv::FileNode fn, int depth=10);

boost::python::dict ORBSlamPython::loadSettingsFile(std::string settingsFilename)
{
    cv::FileStorage fs(settingsFilename.c_str(), cv::FileStorage::READ);
    cv::FileNode root = fs.root();
    if (root.isMap()) 
    {
        return readMap(root);
    }
    else if (root.isSeq())
    {
        boost::python::dict settings;
        settings["root"] = readSequence(root);
        return settings;
    }
    return boost::python::dict();
}


// ----------- HELPER DEFINITIONS -----------
boost::python::dict readMap(cv::FileNode fn, int depth)
{
    boost::python::dict map;
    if (fn.isMap()) {
        cv::FileNodeIterator it = fn.begin(), itEnd = fn.end();
        for (; it != itEnd; ++it) {
            cv::FileNode item = *it;
            std::string key = item.name();
            
            if (item.isNone())
            {
                map[key] = boost::python::object();
            }
            else if (item.isInt())
            {
                map[key] = int(item);
            }
            else if (item.isString())
            {
                map[key] = std::string(item);
            }
            else if (item.isReal())
            {
                map[key] = float(item);
            }
            else if (item.isSeq() && depth > 0)
            {
                map[key] = readSequence(item, depth-1);
            }
            else if (item.isMap() && depth > 0)
            {
                map[key] = readMap(item, depth-1);  // Depth-limited recursive call to read inner maps
            }
        }
    }
    return map;
}

boost::python::list readSequence(cv::FileNode fn, int depth)
{
    boost::python::list sequence;
    if (fn.isSeq()) {
        cv::FileNodeIterator it = fn.begin(), itEnd = fn.end();
        for (; it != itEnd; ++it) {
            cv::FileNode item = *it;
            
            if (item.isNone())
            {
                sequence.append(boost::python::object());
            }
            else if (item.isInt())
            {
                sequence.append(int(item));
            }
            else if (item.isString())
            {
                sequence.append(std::string(item));
            }
            else if (item.isReal())
            {
                sequence.append(float(item));
            }
            else if (item.isSeq() && depth > 0)
            {
                sequence.append(readSequence(item, depth-1)); // Depth-limited recursive call to read nested sequences
            }
            else if (item.isMap() && depth > 0)
            {
                sequence.append(readMap(item, depth-1));
            }
        }
    }
    return sequence;
}
