import numpy as np




def write_ply_xyz_rgb(point_cloud, rgb, output_path):
    point_count=point_cloud.shape[0]
    ply_file = open(output_path, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex " + str(point_count) + "\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")

    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")

    ply_file.write("end_header\n")

    for i in range(point_count):
        ply_file.write(str(point_cloud[i, 0]) + " " +
                       str(point_cloud[i, 1]) + " " +
                       str(point_cloud[i, 2]))

        ply_file.write(" "+str(int(rgb[i, 0])) + " " +
                        str(int(rgb[i, 1])) + " " +
                        str(int(rgb[i, 2])))


        ply_file.write("\n")
    ply_file.close()
    print("save result to "+output_path)
 




def write_ply_xyz(point_cloud,output_path):
    point_count=point_cloud.shape[0]
    ply_file = open(output_path, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex " + str(point_count) + "\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")


    ply_file.write("end_header\n")
    for i in range(point_count):
        ply_file.write(str(point_cloud[i, 0]) + " " +
                       str(point_cloud[i, 1]) + " " +
                       str(point_cloud[i, 2]))

        ply_file.write("\n")
    ply_file.close()
    print("save result to "+output_path)


