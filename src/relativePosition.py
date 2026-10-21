import numpy as np # noqa: I001
import matplotlib.pyplot as plt

from dataStructure import geoLoc, LocalisedDroneImage

class RelativePosition:
    def __init__(self):
        pass

    def getDistanceWithRefImg(self, main : list[LocalisedDroneImage], ref : list[LocalisedDroneImage]):
        if len(main) != len(ref):
            raise ValueError("Length of groundTruth and data must be the same")
        

        self.relativeErrors = []
        self.measuredDistances = []
        self.distancesDrone = []
        self.elevationsDrone = []
        self.poiPairs = []

        for m, r in zip(main, ref):
            for i in range(len(m.objects)//2):
                if i*2 not in r.objects or i*2+1 not in r.objects:
                    continue

                startObjMain = m.objects[i*2]
                startObjRef = r.objects[i*2]
                stopObjMain = m.objects[i*2+1]
                stopObjRef = r.objects[i*2+1]

                mainDist = startObjMain.loc.distTo(stopObjMain.loc)
                # refDist = 1.25
                refDist = startObjRef.loc.distTo(stopObjRef.loc)
                errorDist = np.abs(mainDist - refDist)/refDist * 100
                # dist = np.abs(mainDist - refDist)
            
                # TODO: change to take into account poi elevation
                droneLoc = geoLoc(m.param.GPSLatitude, m.param.GPSLongitude, m.param.GPSAltitudeRelative)
                # elDrone = droneLoc.alt - startObjRef.loc.alt
                
                _, _, hzDistDrone = droneLoc.aer(startObjRef.loc, hzOnly=True)

                self.relativeErrors.append(errorDist)
                self.measuredDistances.append(refDist)
                self.distancesDrone.append(hzDistDrone)
                self.elevationsDrone.append(m.param.GPSAltitudeRelative)


    def getDistance(self, main : list[LocalisedDroneImage], groundMeasurement : dict[str, float]):
       
        self.relativeErrors = []
        self.distancesDrone = []
        self.elevationsDrone = []
        self.poiPairs = []
        self.objPairString = f"{list(groundMeasurement.keys())}"

        for m in main:
            for poiPair, dist in groundMeasurement.items():
                if poiPair == "refPos":
                    continue
                startObjID, stopObjID = poiPair.split("-")
                startObjID = int(startObjID)
                stopObjID = int(stopObjID)

                if startObjID not in m.objects or stopObjID not in m.objects:
                    continue

                startObjMain = m.objects[startObjID]
                stopObjMain = m.objects[stopObjID]

                mainDist = startObjMain.loc.distTo(stopObjMain.loc)
                relErr = np.abs(mainDist - dist)/dist*100
                
                _, _, hzDistDrone = m.param.GPSLoc.aer(groundMeasurement["refPos"], hzOnly=True)

                self.relativeErrors.append(relErr)
                self.distancesDrone.append(hzDistDrone)
                self.elevationsDrone.append(m.param.GPSLoc.alt)
                self.poiPairs.append(poiPair)

    
    def getData(self):
        if hasattr(self, "measuredDistances"):
            return self.distancesDrone, self.elevationsDrone, self.relativeErrors, self.poiPairs, self.measuredDistances
        else:
            return self.distancesDrone, self.elevationsDrone, self.relativeErrors, self.poiPairs

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.distancesDrone, self.elevationsDrone, self.relativeErrors, label='Data')
        ax.set_xlabel('Horizontal Distance')
        ax.set_ylabel('Elevation')
        ax.set_zlabel('Error Distance')
        ax.legend()
        ax.grid(True)
        plt.show()


