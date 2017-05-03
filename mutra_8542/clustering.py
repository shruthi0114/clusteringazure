##  Name: Shruthi Reddy Mutra
##  ID: 1001278542
##  Course: Cloud Computing (CSE-6331)
##  Title: Machine Learning(Kmeans CLustering in Cloud)
##  Reference:  http://glowingpython.blogspot.com/2012/04/k-means-clustering-with-scipy.html
##              http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html
##              http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance.euclidean.html
##              http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.cluster.vq.vq.html#scipy.cluster.vq.vq
##              http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans
##              http://glowingpython.blogspot.com/2012/04/k-means-clustering-with-scipy.html


import csv
import operator
import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import euclidean
from flask import Flask, render_template, request
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/')
def inputs():
    return render_template('input.html')


@app.route('/output/', methods=['GET', 'POST'])
def output():
    x_coordinate = request.form.get('x_coordinate')
    y_coordinate = request.form.get('y_coordinate')
    no_clusters = int(request.form.get('n_clusters'))

    column_id = {1: 'time', 2: 'latitude', 3: 'longitude', 4: 'depth', 5: 'mag', 6: 'magType', 7: 'nst', 8: 'gap',
                 9: 'dmin', 10: 'rms', 11: 'net', 12: 'id', 13: 'updated', 14: 'place', 15: 'type'}

    with open('all_month.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = []
        data = []
        i = 0
        for r in reader:
            if i == 0:
                header = r

            else:
                data.append([(r[header.index(column_id.get(int(x_coordinate)))]) or float(0.0),
                             r[header.index(column_id.get(int(y_coordinate)))] or int(0)])
            i = i + 1
        with open('output.csv', 'w') as result:
            writer = csv.writer(result, delimiter=',')
            for r in data:
                writer.writerow([r[0], r[1]])

        data = np.array(data)
        data = data.astype(float)
        # print _data
        centroids, distort = kmeans(data, no_clusters)
        centroids_list = centroids.tolist()
        print('\nCENTROIDS:\n')
        for c in centroids_list:
            print(c)

        idx, _ = vq(data, centroids)

        euclid = {}
        print('\nDistance Between Centroids \n')
        for c in centroids_list:
            for e in centroids_list[centroids_list.index(c) + 1:]:
                euclid[str(centroids_list.index(c)) + ' and ' + str(centroids_list.index(e))] = euclidean(c, e)
        euclid = sorted(euclid.items(), key=operator.itemgetter(0))
        print(euclid)
        cnt_idx = idx.tolist()
        dict_index = {}

        print('\nNumber of points in each cluster\n')
        for i in cnt_idx:
            dict_index[i] = cnt_idx.count(i)
        print(dict_index)
        with open('barchart.csv', 'w') as barchart:
            barwriter = csv.writer(barchart, delimiter=',')
            barwriter.writerow(['Cluster', 'Point'])
            for k, v in dict_index.items():
                barwriter.writerow([k, v])
        color = ["r.", "g.", "b.", "y.", "k.", "b.", "m.", "c."]
        for n in range(no_clusters):
            plt.plot(data[idx == n, 0], data[idx == n, 1], color[n % 8], marker="x", markersize=5)
        plt.plot(centroids[:, 0], centroids[:, 1], "sm", markersize=5)
        plt.savefig("static\image.jpg")
        plt.clf()
        print(len(dict_index))
        print(dict_index.values())
        plt.bar(range(len(dict_index)), dict_index.values(), width=1 / 1.5, color="blue")
        plt.xticks(range(len(dict_index)), list(dict_index.keys()))
        plt.savefig("static\image_bar.jpg")

    return render_template("output.html", dict_index=dict_index, centroids_list=centroids_list)


if __name__ == "__main__":
    app.debug = True
    app.run("")
