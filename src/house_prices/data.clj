(ns house-prices.data
  (:require [incanter.charts :as c]
            [incanter.core :as i]
            [incanter.stats :as s]
            [incanter.io :as iio]))

;; Data In / Out

;; File Paths
(def file-paths {:train       "./data/train.csv"
                 :test        "./data/test.csv"
                 :predictions "./data/predictions.csv"})

(defn load-data
  "Reads in data from file."
  [file-path]
  (iio/read-dataset file-path
                    :header true
                    :keyword-headers true))

(defn save-data
  "Saves data to a file."
  [file-path data]
  (i/save data file-path))

(defn save-predictions
  "Save the predictions to a .csv file"
  [predictions]
  (save-data (:predictions file-paths) predictions))

;; Dataset manipulation#

(defn mean-column
  "Adds a derived column by replacing non-numeric (NA) values
  in the source column with the mean."
  [to-column from-column dataset]
  (let [col-data (i/$ from-column dataset)
        mean     (s/mean (filter number? col-data))
        f        #(if (number? %)
                    %
                    mean)]
    (i/add-derived-column to-column [from-column] f dataset)))


