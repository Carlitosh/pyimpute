#!/usr/bin/env python3

from _main import load_training_vector, load_targets, impute, evaluate_clf
from sklearn.ensemble import RandomForestClassifier
import os



basedir = 'C:/Users/carli/Documents/Proyectos_Desarrollo_de_Software/pyimpute/src/pyimpute'

feb18 = os.path.join(basedir, 'feb18.tif')
nov17 = os.path.join(basedir, 'nov17.tif')
jun18 = os.path.join(basedir, 'jun18.tif')
sep18 = os.path.join(basedir, 'sep18.tif')


#================================================================
explanatory_rasters = [feb18 , nov17 , jun18 , sep18]


response_data = 'train_puntos.geojson'

train_xs, train_y = load_training_vector(response_data,
                                         explanatory_rasters,
                                         response_field="amb")

#================================================================


clf = RandomForestClassifier(n_estimators=10, n_jobs=1)
clf.fit(train_xs, train_y) 

#=================================================================

evaluate_clf(clf, train_xs, train_y)

#==================================================================

target_xs, raster_info = load_targets(explanatory_rasters)


#==================================================================



impute(target_xs, clf, raster_info, outdir='./tmp',
        linechunk=400, class_prob=True, certainty=True)

