'''
Created on 20 mrt. 2013

@author: Erik Vandeputte
'''
import kmeans_assign
import regression
import time

RESULTSFILE = "results.txt"

clusters =  [100,200,250,300,400,450,500,700]
start_time = time.time()
f = open(RESULTSFILE,'a')
for num_clusters in clusters:
    print 'perform kmeans for %d clusters' %num_clusters
    #kmeans_full.main(num_clusters)
    #assign (hard and soft)
    kmeans_assign.main(True,num_clusters)
    kmeans_assign.main(False,num_clusters)
    #perform regression(hard and soft)
    mse_hard,alpha_hard = regression.main(True, num_clusters)
    mse_soft,alpha_soft = regression.main(False, num_clusters)
    f.write("%s\t%s\t%s\t%s\n" % (str(num_clusters),'hard',str(mse_hard),str(alpha_hard)))
    f.write("%s\t%s\t%s\t%s\n" % (str(num_clusters),'soft',str(mse_soft),str(alpha_soft)))
    f.flush()
f.write("running this script took %.2f seconds" % (time.time() - start_time))
f.close()