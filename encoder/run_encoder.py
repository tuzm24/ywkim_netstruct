from threading import Thread
import subprocess
import time
import queue
import ruamel.yaml
import logging
import os
import sys


Sequences = """Name,FileName,BitDepth,Format,FrameRate,0,width,height,Frames,Level
AerialCrowd_3840x2160_30,AerialCrowd_3840x2160_30_10b_709_420.yuv,10,420,30,0,3840,2160,600,5.1
BeachMountain2_3840x2160_30fps,BeachMountain2_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
BeachMountain_3840x2160_30fps,BeachMountain_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
BridgeViewTraffic_3840x2160_60,BridgeViewTraffic_3840x2160_60_10b_709_420.yuv,10,420,60,0,3840,2160,600,5.1
BuildingHall1_3840x2160_50fps,BuildingHall1_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,500,5.1
BuildingHall_3840x2160_50fps,BuildingHall_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
BundNightscape_3840x2160_30fps,BundNightscape_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
ConstructionField_3840x2160_30fps,ConstructionField_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
CrossRoad1_3840x2160_50fps,CrossRoad1_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
CrossRoad2_3840x2160_50fps,CrossRoad2_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
CrossRoad3_3840x2160_50fps,CrossRoad3_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
Crosswalk1_4096x2160_60fps,Crosswalk1_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,470,5.1
DayStreet_3840x2160_60p,DayStreet_3840x2160_60p_10bit_420_hlg.yuv,10,420,60,0,3840,2160,600,5.1
DinningHall2_3840x2160_50fps,DinningHall2_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
DroneTakeOff_3840x2160_30fps,DroneTakeOff_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
FlyingBirds2_3840x2160p_60,FlyingBirds2_3840x2160p_60_10b_HLG_420.yuv,10,420,60,0,3840,2160,300,5.1
FlyingBirds_3840x2160p_60,FlyingBirds_3840x2160p_60_10b_HLG_420.yuv,10,420,60,0,3840,2160,600,5.1
Fountains_3840x2160_30fps,Fountains_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
IceAerial_3840x2160_30fps,IceAerial_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
IceRiver_3840x2160_30fps,IceRiver_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
IceRock2_3840x2160_30fps,IceRock2_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
IceRock_3840x2160_30fps,IceRock_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
Library_3840x2160_30fps,Library_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
Marathon_3840x2160_30fps,Marathon_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
Metro_3840x2160_60,Metro_3840x2160_60_10b_709_420.yuv,10,420,60,0,3840,2160,600,5.1
MountainBay2_3840x2160_30fps,MountainBay2_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
MountainBay_3840x2160_30fps,MountainBay_3840x2160_30fps_420_10bit.yuv,10,420,30,0,3840,2160,300,5.1
Netflix_Aerial_4096x2160_60fps,Netflix_Aerial_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_BarScene_4096x2160_60fps,Netflix_BarScene_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_Boat_4096x2160_60fps,Netflix_Boat_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,300,5.1
Netflix_BoxingPractice_4096x2160_60fps,Netflix_BoxingPractice_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,254,5.1
Netflix_Crosswalk_4096x2160_60fps,Netflix_Crosswalk_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,300,5.1
Netflix_Dancers_4096x2160_60fps,Netflix_Dancers_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_DinnerScene_4096x2160_60fps,Netflix_DinnerScene_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_DrivingPOV_4096x2160_60fps,Netflix_DrivingPOV_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_FoodMarket2_4096x2160_60fps,Netflix_FoodMarket2_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,300,5.1
Netflix_FoodMarket_4096x2160_60fps,Netflix_FoodMarket_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,600,5.1
Netflix_Narrator_4096x2160_60fps,Netflix_Narrator_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,300,5.1
Netflix_PierSeaside_4096x2160_60fps,Netflix_PierSeaside_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_RitualDance_4096x2160_60fps,Netflix_RitualDance_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,600,5.1
Netflix_RollerCoaster_4096x2160_60fps,Netflix_RollerCoaster_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_SquareAndTimelapse_4096x2160_60fps,Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,600,5.1
Netflix_TimeLapse_4096x2160_60fps,Netflix_TimeLapse_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,600,5.1
Netflix_ToddlerFountain_4096x2160_60fps,Netflix_ToddlerFountain_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
Netflix_TunnelFlag_4096x2160_60fps,Netflix_TunnelFlag_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,600,5.1
Netflix_WindAndNature_4096x2160_60fps,Netflix_WindAndNature_4096x2160_60fps_10bit_420.yuv,10,420,60,0,4096,2160,1199,5.1
NightRoad_3840x2160_60,NightRoad_3840x2160_60_10b_709_420.yuv,10,420,60,0,3840,2160,600,5.1
ParkLake_3840x2160_50fps,ParkLake_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
ParkRunning1_3840x2160_50fps,ParkRunning1_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
PeopleInShoppingCenter_3840x2160_60p,PeopleInShoppingCenter_3840x2160_60p_10bit_420_hlg.yuv,10,420,60,0,3840,2160,600,5.1
ResidentialBuilding_3840x2160_30fps,ResidentialBuilding_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
ResidentialGate1_3840x2160_50fps,ResidentialGate1_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,1000,5.1
Runners_3840x2160_30fps,Runners_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
RushHour_3840x2160_30fps,RushHour_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
Scarf_3840x2160_30fps,Scarf_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
Square_3840x2160_60,Square_3840x2160_60_10b_709_420.yuv,10,420,60,0,3840,2160,600,5.1
SunsetBeach_3840x2160p_60,SunsetBeach_3840x2160p_60_10b_HLG_420.yuv,10,420,60,0,3840,2160,600,5.1
TallBuildings_3840x2160_30fps,TallBuildings_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
TrafficAndBuilding_3840x2160_30fps,TrafficAndBuilding_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
TrafficFlow_3840x2160_30fps,TrafficFlow_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
TreeShade_3840x2160_30fps,TreeShade_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
Wood_3840x2160_30fps,Wood_3840x2160_30fps_10bit_420.yuv,10,420,30,0,3840,2160,300,5.1
Cosmos1_1920x856_24fps,Cosmos1_1920x856_24fps_420.yuv,8,420,24,0,1920,856,480,5.1
Fountains_1920x1080_30fps,Fountains_1920x1080_30fps_10bit_420.yuv,10,420,30,0,1920,1080,300,5.1
FreeSardines1_1920x1080_120fps,FreeSardines1_1920x1080_120fps_10bit_420.yuv,10,420,120,0,1920,1080,600,5.1
Hurdles_1920x1080p_50,Hurdles_1920x1080p_50_10b_pq_709_ct2020_420_rev1.yuv,10,420,50,0,1920,1080,500,5.1
IceAerial_1920x1080_30fps,IceAerial_1920x1080_30fps_420_10bit.yuv,10,420,30,0,1920,1080,300,5.1
IceRiver_1920x1080_30fps,IceRiver_1920x1080_30fps_420_10bit.yuv,10,420,30,0,1920,1080,300,5.1
IceRock2_1920x1080_30fps,IceRock2_1920x1080_30fps_420_10bit.yuv,10,420,30,0,1920,1080,300,5.1
IceRock_1920x1080_30fps,IceRock_1920x1080_30fps_420_10bit.yuv,10,420,30,0,1920,1080,300,5.1
Market3_1920x1080p_50,Market3_1920x1080p_50_10b_pq_709_ct2020_420_rev1.yuv,10,420,50,0,1920,1080,400,5.1
Metro_1920x1080_60fps,Metro_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,600,5.1
Netflix_Aerial_1920x1080_60fps,Netflix_Aerial_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,1199,5.1
Netflix_BarScene_1920x1080_60fps,Netflix_BarScene_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,1199,5.1
Netflix_Crosswalk_1920x1080_60fps,Netflix_Crosswalk_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,300,5.1
Netflix_DrivingPOV_1920x1080_60fps,Netflix_DrivingPOV_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,1199,5.1
Netflix_FoodMarket2_1920x1080_60fps,Netflix_FoodMarket2_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,300,5.1
Netflix_FoodMarket_1920x1080_60fps,Netflix_FoodMarket_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,600,5.1
Netflix_PierSeaside_1920x1080_60fps,Netflix_PierSeaside_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,1199,5.1
Netflix_RitualDance_1920x1080_60fps,Netflix_RitualDance_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,600,5.1
Netflix_SquareAndTimelapse_1920x1080_60fps,Netflix_SquareAndTimelapse_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,600,5.1
Netflix_Timelapse_1920x1080_60fps,Netflix_Timelapse_1920x1080_60fps_10bit_420_CfE.yuv,10,420,60,0,1920,1080,600,5.1
Netflix_WindAndNature_1920x1080_60fps,Netflix_WindAndNature_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,1199,5.1
Rowing2_1920x1080_120fps,Rowing2_1920x1080_120fps_10bit_420.yuv,10,420,120,0,1920,1080,600,5.1
Runners_1920x1080_30fps,Runners_1920x1080_30fps_10bit_420.yuv,10,420,30,0,1920,1080,300,5.1
RushHour_1920x1080_30fps,RushHour_1920x1080_30fps_10bit_420.yuv,10,420,30,0,1920,1080,300,5.1
SakuraGate_1920x1080_60,SakuraGate_1920x1080_60_8bit.yuv,8,420,60,0,1920,1080,600,5.1
ShowGirl2TeaserClip4000_1920x1080p_24,ShowGirl2TeaserClip4000_1920x1080p_24_10bit_12_P3_ct2020_rev1.yuv,10,420,24,0,1920,1080,339,5.1
Starting_1920x1080p_50,Starting_1920x1080p_50_10b_pq_709_ct2020_420_rev1.yuv,10,420,50,0,1920,1080,500,5.1
BasketballDrillText_832x480_50,BasketballDrillText_832x480_50.yuv,8,420,50,0,832,480,500,5.1
BasketballDrill_832x480_50,BasketballDrill_832x480_50.yuv,8,420,50,0,832,480,500,5.1
BasketballDrive_1920x1080_50,BasketballDrive_1920x1080_50.yuv,8,420,50,0,1920,1080,500,5.1
BasketballPass_416x240_50,BasketballPass_416x240_50.yuv,8,420,50,0,416,240,500,5.1
BlowingBubbles_416x240_50,BlowingBubbles_416x240_50.yuv,8,420,50,0,416,240,500,5.1
BQMall_832x480_60,BQMall_832x480_60.yuv,8,420,60,0,832,480,600,5.1
BQSquare_416x240_60,BQSquare_416x240_60.yuv,8,420,60,0,416,240,600,5.1
BQTerrace_1920x1080_60,BQTerrace_1920x1080_60.yuv,8,420,60,0,1920,1080,600,5.1
Cactus_1920x1080_50,Cactus_1920x1080_50.yuv,8,420,50,0,1920,1080,500,5.1
Campfire_3840x2160_30fps,Campfire_3840x2160_30fps_10bit_420_bt709_videoRange.yuv,10,420,30,0,3840,2160,300,5.1
CatRobot1_3840x2160p_60,CatRobot1_3840x2160p_60_10_709_420.yuv,8,420,60,0,3840,2160,1200,5.1
ChinaSpeed_1024x768_30,ChinaSpeed_1024x768_30.yuv,8,420,30,0,1024,768,500,5.1
DaylightRoad2_3840x2160_60fps,DaylightRoad2_3840x2160_60fps_10bit_420.yuv,10,420,60,0,3840,2160,600,5.1
FoodMarket4_3840x2160_60fps,FoodMarket4_3840x2160_60fps_10bit_420.yuv,10,420,60,0,3840,2160,720,5.1
FourPeople_1280x720_60,FourPeople_1280x720_60.yuv,8,420,60,0,1280,720,600,5.1
Johnny_1280x720_60,Johnny_1280x720_60.yuv,8,420,60,0,1280,720,600,5.1
Kimono1_1920x1080_24,Kimono1_1920x1080_24.yuv,8,420,24,0,1920,1080,240,5.1
KristenAndSara_1280x720_60,KristenAndSara_1280x720_60.yuv,8,420,60,0,1280,720,600,5.1
MarketPlace_1920x1080_60fps,MarketPlace_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,600,5.1
NebutaFestival_2560x1600_60,NebutaFestival_2560x1600_60_10bit_crop.yuv,10,420,60,0,2560,1600,300,5.1
ParkRunning3_3840x2160_50fps,ParkRunning3_3840x2160_50fps_10bit_420.yuv,10,420,50,0,3840,2160,500,5.1
ParkScene_1920x1080_24,ParkScene_1920x1080_24.yuv,8,420,24,0,1920,1080,240,5.1
PartyScene_832x480_50,PartyScene_832x480_50.yuv,8,420,50,0,832,480,500,5.1
PeopleOnStreet_2560x1600_30,PeopleOnStreet_2560x1600_30_crop.yuv,8,420,30,0,2560,1600,150,5.1
RaceHorses_416x240_30,RaceHorses_416x240_30.yuv,8,420,30,0,416,240,300,5.1
RaceHorses_832x480_30,RaceHorses_832x480_30.yuv,8,420,30,0,832,480,300,5.1
RitualDance_1920x1080_60fps,RitualDance_1920x1080_60fps_10bit_420.yuv,10,420,60,0,1920,1080,600,5.1
SlideEditing_1280x720_30,SlideEditing_1280x720_30.yuv,8,420,30,0,1280,720,300,5.1
SlideShow_1280x720_20,SlideShow_1280x720_20.yuv,8,420,20,0,1280,720,500,5.1
SteamLocomotiveTrain_2560x1600_60,SteamLocomotiveTrain_2560x1600_60_10bit_crop.yuv,10,420,60,0,2560,1600,300,5.1
Tango2_3840x2160_60fps,Tango2_3840x2160_60fps_10bit_420.yuv,10,420,60,0,3840,2160,294,5.1
TrafficFlow_3840x2160_30fps,TrafficFlow_3840x2160_30fps_10bit_420_jvet.yuv,10,420,30,0,3840,2160,300,5.1
"""






def get_str_time():
    return time.strftime('%a %d %b %Y, %Hh%Mm%S', time.localtime(time.time()))

class LoggingHelper(object):
    INSTANCE = None
    p_start_time = time.time()
    # if LoggingHelper.INSTANCE is not None:
    #     raise ValueError("An instantiation already exists!")

    os.makedirs("./logs", exist_ok=True)
    logger = logging.getLogger()

    logging.basicConfig(filename='./logs/'+ get_str_time() + '_LOGGER_basic.log', level=logging.INFO)

    fileHandler = logging.FileHandler("./logs/" + get_str_time() + '_msg.log')
    streamHandler = logging.StreamHandler()

    fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    fileHandler.setFormatter(fomatter)
    streamHandler.setFormatter(fomatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)


    def __init__(self):
        pass
    @classmethod
    def get_instance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = LoggingHelper()
        return cls.INSTANCE

    @staticmethod
    def diff_time_logger(messege, start_time):
        LoggingHelper.get_instance().logger.info("[{}] :: running time {}".format(messege, time.time() - start_time))



    def log_cur_time(self):
        self.logger.info("Clock : %s", self.get_str_time())

    def log_diff_start_time(self):
        self.logger.info("Elapsed Time : %a %m %b %Y, %Hh%Mm%S", time.localtime(time.time() - self.p_start_time))

class ConfigMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

class Config(dict):
    yaml = ruamel.yaml.YAML()
    yaml.allow_duplicate_keys = True

    def __init__(self, file_path, logger):
        # print(os.getcwd())
        assert os.path.exists(file_path), "ERROR: Config File doesn't exist."
        with open(file_path, 'r') as f:
            self.member = self.yaml.load(f)
            f.close()
        self.logger = logger

    def __getattr__(self, name):
        if name not in self.member:
            if self.logger is None:
                print("Miss no name '%s' in config ", name)
            else:
                self.logger.error("Miss no name '%s' in config ", name)
            return False
        value = self.member[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

    def isExist(self, name):
        if name in self.member:
            return True
        return False


    def write_yml(self):
        path = self.member['NET_INFO_PATH']
        with open(path, 'w+') as fp:
            self.yaml.dump(self.member, fp)



class SequnceInfo(object):
    #0: Name,    1: FileName,   2: BitDepth,    3: Format,  4: FrameRate,   5: 0,   6: width,   7:height,   8:Frames,   9:Level (minus 1)
    def __init__(self, info):
        self.name = info[0]
        self.yuvName = info[1]
        self.bitdepth = info[2]
        self.format = info[3]
        self.frameRate = info[4]
        self.zero = info[5]
        self.width = info[6]
        self.height = info[7]
        self.numframe = info[8]
        self.level = info[9]
        self.frameskip = 0



class RunEncoder(object):
    orgpath = ""
    encpath = "./EncoderApp.exe"
    cndpath = "./cfg"
    corenum = os.cpu_count()
    if corenum>32:
        corenum = 32
    corenum -= 2
    if corenum<0:
        corenum = 1
    temppath = "./temp"
    binpath = "./bin"
    logpath = "./log"
    tobeEncodeFrame = 300
    os.makedirs(temppath, exist_ok=True)
    os.makedirs(binpath, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)
    Sequence_Path = 'C:/seq'
    # Sequence_Path = 'C:/origCfP'
    qps = ['22', '27', '32', '37']


    def __init__(self):
        self.runningcore = 0
        self.logger = LoggingHelper.get_instance().logger
        self.cndpaths = self.getFileList(self.cndpath, '.cfg')
        self.frameNum = RunEncoder.tobeEncodeFrame



    def getFileList(self, dir, pattern='.yuv'):
        matches = []
        # for root, dirnames, filenames in os.walk(dir):
        #     for filename in filenames:
        for filename in os.listdir(dir):
            if filename.endswith(pattern):
                matches.append(os.path.join(dir, filename))
        return matches


    def initSeqeuences(self):
        candis = self.getFileList(RunEncoder.Sequence_Path)
        # names = []
        # for candi in candis:
        #     candi = os.path.basename(candi)
        #     candi = candi.split("_")[0]
        #     names.append(candi.lower())
        # seqs = pd.read_csv("./CTCSequences.csv", sep=",")
        # seqs = seqs.as_matrix()
        seqs = []
        # with open("./Sequences.csv", 'r') as reader:
        # with open("./SequenceSetting/CTCSequences.csv", 'r') as reader:
        #     data = reader.read()
        lines = Sequences.strip().split('\n')
        for line in lines:
            seqs.append(line.split(','))
        seqs = seqs[1:]
        seqlist = {}
        #format
        #0: Name,    1: FileName,   2: BitDepth,    3: Format,  4: FrameRate,   5: 0,   6: width,   7:height,   8:Frames,   9:Level (minus 1)
        # for seq in seqs:
        for candi in candis:
            if os.path.basename(candi).split("_")[0].lower() != "netflix":
                tmp = '_'.join(os.path.basename(candi).split("_")[:3]).lower()
            else:
                tmp = '_'.join(os.path.basename(candi).split("_")[:4]).lower()
            # print(seq[0], tmp)
            for seq in seqs:
                if seq[0].lower()== tmp:
                    seqlist[candi] = seq
                    break
            else:
                self.logger.error('[No Sequence Infomation : %s]' %os.path.basename(candi))
                exit()
                    #seqdic[seq[0]] = seq[1:]
        return seqlist


    def runProcess(self, enc_name,  command, logpath):
        print("%s start" %enc_name)
        self.runningcore +=1
        with open(logpath, 'w') as fp:
            sub_proc = subprocess.Popen(command, stdout = fp)
            sub_proc.wait()
        print("%s finished" %enc_name)
        self.runningcore -=1
        return

    def RunEncoder(self):
        seqlist = self.initSeqeuences()
        self.taskqueue = queue.Queue()
        for cnd in self.cndpaths:
            config = Config(cnd, self.logger)
            for qp in self.qps:
                for key, value in seqlist.items():
                    number_frame = self.getEncodeFrame(seq = value)
                    config.IntraPeriod = intra_Preriod = self.getIntraPeriod(value[4], config)
                    if intra_Preriod<2:
                        enc_name, command, logpath = self.get_enc_comand(key, value, qp, cnd, config, -1, -1, number_frame)
                        self.taskqueue.put((enc_name, command, logpath))
                    else:
                        number_ras = self.getRasNum(intraPeriod=intra_Preriod, tobeEncodeFrame=number_frame)
                        for rid in range(number_ras):
                            enc_name, command, logpath = self.get_enc_comand(key, value, qp, cnd, config, rid, number_ras, number_frame)
                            self.taskqueue.put((enc_name, command, logpath))
        print(list(self.taskqueue.queue))
        self.logger.info("[Number of Task %s]" %self.taskqueue.qsize())
        self.logger.info("[Number of Cpu Core Setting %s]" %self.corenum)
        yandn = input("[Do you want to proceed? [y/n]  ")
        if yandn.lower() != 'y':
            self.logger.info("Exit")
            sys.exit()
        total_num = self.taskqueue.qsize()
        q_count = 0

        while self.taskqueue.qsize() != 0:
            if self.runningcore>=RunEncoder.corenum:
                time.sleep(1)
                continue
            enc_name, command, logpath = self.taskqueue.get()
            t = Thread(target=self.runProcess, args=(enc_name, command, logpath,))
            t.start()
            self.logger.info("[%s/%s]"%(q_count, total_num))
            q_count +=1

    def make_seqcfg_file(self, filepath, yuvpath, seq, cndcfg, rid, rasnum, tobeEncodeFrame):
        #0: Name,    1: FileName,   2: BitDepth,    3: Format,  4: FrameRate,   5: 0,   6: width,   7:height,   8:Frames,   9:Level (minus 1)
        image_path = yuvpath
        ip = cndcfg.IntraPeriod
        frameskip = 0
        if rid>=0:
            frameskip = rid * ip
            if rid==rasnum-1:
                tobeEncodeFrame -= rid * ip
            else:
                tobeEncodeFrame = ip + 1
        with open(filepath, 'w') as f:
            f.write("InputFile : %s\n" %(image_path))
            f.write("InputBitDepth : %s\n" %(seq[2]))
            f.write("InputChromaFormat : %s\n" %(seq[3]))
            f.write("FrameRate : %s\n" %(seq[4]))
            # f.write("FrameSkip : %s\n" %(taskdata[9]))
            f.write("FrameSkip : %s\n" %(frameskip))
            f.write("SourceWidth : %s\n" %(seq[6]))
            f.write("SourceHeight : %s\n" %(seq[7]))
            f.write("FramesToBeEncoded : %s\n" %(tobeEncodeFrame))
            level = float(seq[9])
            if level%1==0:
                level = int(level)
            f.write("Level : %s\n" %(level))

    def getCndName(self, cfgfile):
        cfg = Config(cfgfile, self.logger)
        ip = cfg.IntraPeriod
        if ip == 1:
            return 'AI'
        elif ip < 0:
            if ((cfg.Frame1)[0] == 'B'):
                return 'LDB'
            else:
                return 'LDP'
        else:
            return 'RA'

    def getRasNum(self, intraPeriod, tobeEncodeFrame):
        if intraPeriod < 2:
            return -1
        return (tobeEncodeFrame + intraPeriod - 1) // intraPeriod

    def getIntraPeriod(self, frameRate, cfg):
        intra_period = {20:16, 24:32, 30:32, 50:48, 60:64, 100:96, 120:112}
        # if cfg.IntraPeriod<0:
        #     if((cfg.Frame1)[0]=='B'):
        #         return -1
        #     else:
        #         return -2
        if cfg.IntraPeriod<2:
            return cfg.IntraPeriod
        return intra_period[int(frameRate)]

    def getEncodeFrame(self, seq):
        if self.frameNum==0:
            return int(seq[8])
        if self.frameNum>0:
            return min([int(seq[8]), self.frameNum])
        else:
            return min([-self.frameNum*seq.frameRate, int(seq[8])])

    def get_enc_comand(self, yuvpath, seq, qp, cnd, cfg, rid, rasnum, tobeEncodedFrame):
        #0: Name,    1: FileName,   2: BitDepth,    3: Format,  4: FrameRate,   5: 0,   6: width,   7:height,   8:Frames,   9:Level (minus 1)
        taskname = seq[0]
        # cndname = taskdata[2]
        if rid>=0:
            str_rid = '_RS' + str(rid)
        else:
            str_rid = ''
        str_bitdepth = '_' + seq[2] + 'bit'
        enc_logpath = self.logpath + '/' + 'enc_' + taskname + '_' + qp + str_rid + '.log'
        enc_path = self.encpath
        cnd_name = self.getCndName(cnd)
        seqcfg_path = self.temppath + '/' + taskname + str_rid + '.cfg'
        bin_path = self.binpath + '/' + cnd_name + '_' + taskname + str_bitdepth + '_' + qp + str_rid + '.bin'
        enc_name = cnd_name + '_' + taskname+ '_' + qp + str_rid
        enc_command = enc_path + ' -c ' + cnd + ' -c ' + seqcfg_path + ' -q ' + qp + ' -b ' + bin_path
        # if int(taskdata[4]) > 1: # intra Period
            # enc_command += ' -ip ' + taskdata[4]
        self.make_seqcfg_file(seqcfg_path, yuvpath, seq, cfg, rid, rasnum, tobeEncodedFrame)
        return enc_name, enc_command, enc_logpath


if __name__=='__main__':
    r = RunEncoder()
    r.RunEncoder()
    print("Done")