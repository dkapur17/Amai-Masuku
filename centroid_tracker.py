from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():
    def __init__(self, maxDisappeared=20):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.frames = 0
        self.entry_exit_data = []
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        self.entry_exit_data.append((self.frames/30, -1, True))

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        self.entry_exit_data[objectID] = (
            self.entry_exit_data[objectID][0], self.frames/30,
            self.entry_exit_data[objectID][2])

    def update(self, boxes, mask_stats):
        '''main updation of tracking'''
        id_maps = []

        if len(boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(boxes), 2), dtype="int")
        for (i, (startX, startY, width, height)) in enumerate(boxes):
            cX = int((2 * startX + width) / 2.0)
            cY = int((2 * startY + height) / 2.0)
            inputCentroids[i] = (cX, cY)
            id_maps.append(
                {"centroid": (cX, cY), "mask": mask_stats[i], "id": i})

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

            for key in self.objects.keys():
                centroid = self.objects[key]
                id = -1
                for mp in id_maps:
                    if mp['centroid'][0] == centroid[0] and mp['centroid'][1] == centroid[1]:
                        id = mp['id']
                if id == -1:
                    continue
                self.entry_exit_data[key] = (self.entry_exit_data[key][0],
                                             self.entry_exit_data[key][1],
                                             self.entry_exit_data[key][2] and mask_stats[id])

        self.frames += 1
        return self.objects

    def write_file(self, file_name):
        f = open(file_name, 'w')
        for i, entry in enumerate(self.entry_exit_data):
            f.write(
                f'{i + 1}. Entry: {entry[0]:.2f}s, Exit:  {round(entry[1], 2) if entry[1] > 0 else "---"}s, Mask: {["No","Yes"][entry[2]]}\n')
