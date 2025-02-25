import os



seqs = """AerialCrowd_3840x2160_30,AerialCrowd_3840x2160_30_10b_709_420.yuv,10,420,30,0,3840,2160,600,5.1
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
TrafficFlow_3840x2160_30fps,TrafficFlow_3840x2160_30fps_10bit_420_jvet.yuv,10,420,30,0,3840,2160,300,5.1"""

class ENUM_TARGET(object):
    UHD = 0
    FHD = 1
    CTC = 2

TARGET = ENUM_TARGET.UHD

seqs = seqs.split('\n')

for i, f in enumerate(seqs):
    seqs[i] = seqs[i].split(',')

if TARGET == ENUM_TARGET.UHD:
    seqs = seqs[:62]
    dirs = ['./training', './validation']
elif TARGET == ENUM_TARGET.FHD:
    seqs = seqs[62:90]
    dirs = ['./FHD/training', './FHD/validation']
else:
    seqs = seqs[90:]
    dirs = ['./test']



for dir in dirs:
    for filename in os.listdir(dir):
        if filename.endswith('.bin'):
            targetname = filename.split('_')
            for seq in seqs:
                if targetname[1].lower() == seq[0].split('_')[0].lower():

                    if targetname[1].lower() == 'netflix':
                        if targetname[2].lower() == seq[0].split('_')[1].lower():
                            targetname[2] = '_'.join(seq[0].split('_')[1:])
                            newname = '_'.join(targetname)
                            os.rename(os.path.join(dir, filename), os.path.join(dir, newname))
                            break
                    else:
                        targetname[1] = seq[0]
                        newname = '_'.join(targetname)
                        os.rename(os.path.join(dir, filename), os.path.join(dir, newname))
                        break
                else:
                    print(filename)

