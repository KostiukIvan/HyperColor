import numpy as np
import os.path


class Mesh:

    def __init__(self, path, n_points):
        self.list_of_meshes = []
        self.data_dir = None
        self.points_range = [2048, 4096]
        self.n_points = None
        self.data_dir = path
        self.n_points = n_points
        self.load()

    def load(self):
        index = 0
        while True:
            sphere_path = self.data_dir + "/sphere_" + str(index) + "_" + str(self.n_points) +".txt"
            
            if not os.path.isfile(sphere_path):
                break
            
            point_cloud = np.fromfile(sphere_path, float, -1, ' ')
            point_cloud = point_cloud.reshape(self.n_points, 3)

            faces_path = self.data_dir + "/mesh_" + str(index) + "_" + str(self.n_points) +".txt"
            
            if not os.path.isfile(faces_path):
                break

            faces = np.fromfile(faces_path, float, -1, ' ')
            faces.astype(int)
            faces = faces.reshape(int(faces.size/3), 3) 

            index += 1

            #store data
            self.list_of_meshes.append((point_cloud,faces))

        if len(self.list_of_meshes) == 1000:
            print("All meshes loaded!!!")
        else:
            print(f"ERROR: Model loaded {len(self.list_of_meshes)} meshes")
            
    def get_random_object(self):
        id = np.random.randint(len(self.list_of_meshes))
        return self.list_of_meshes[id]


    def get_item(self, id):
        if id < 0 or id >= len(self.list_of_meshes):
            return -1

        return self.list_of_meshes[id]

        

