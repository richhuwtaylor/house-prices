(ns house-prices.core
  (:require [clj-ml.classifiers :refer [make-classifier
                                        classifier-train
                                        classifier-evaluate
                                        classifier-classify
                                        classifier-predict-numeric]]
            [clj-ml.data :as clj-ml-data]
            [clj-ml.filters :refer [make-apply-filter]]
            [clj-ml.io :refer [load-instances]]
            [house-prices.data :as data]
            [house-prices.stats :as stats]
            [incanter.core :as i]
            [incanter.stats :as s]))

(defn feature-matrix
  "Select specific columns from the given dataset as a matrix."
  [col-names dataset]
  (-> (i/$ col-names dataset)
      (i/to-matrix)))

(defn add-bias
  [x]
  "Adds a bias term column as the first column to the supplied matrix."
  (i/bind-columns (repeat (i/nrow x) 1) x))

(defn train-model
  "Trains the model for the given test data, returns a matrix of coefficients."
  [train-data features]
  (let [x    (->> train-data
                  (feature-matrix features)
                  (add-bias))
        y    (->> (i/$ :SalePrice train-data)
                  (i/matrix))
        beta (stats/normal-equation x y)]
    beta))

(defn test-matrix
  "Returns a matrix from the dataset with the selected features as columns."
  [test-data features]
  (->> test-data
       (feature-matrix features)
       (add-bias)))

(defn predict-house-prices
  "Uses a ordinary least squares linear regression model trained on the training
   set to make and save predictions on the test set."
  []
  (let [selected-features [:LotArea
                           :OverallQual
                           :OverallCond
                           :LotFrontageMean]
        train-data        (->> (data/load-data (:train data/file-paths))
                               (data/mean-column :LotFrontageMean :LotFrontage))
        test-data         (->> (data/load-data (:test data/file-paths))
                               (data/mean-column :LotFrontageMean :LotFrontage))
        ids               (i/$ 0 test-data)
        ys                (->> (i/$ :SalePrice train-data)
                               (i/matrix))
        beta              (train-model train-data selected-features)
        test-matrix       (test-matrix test-data selected-features)
        results           (i/mmult test-matrix beta)
        predictions       (->> (i/matrix [ids results])
                               (i/trans)
                               (i/dataset ["Id" "SalePrice"]))]
    (data/save-predictions predictions)))

(defn train-classifier
  []
  (let [train-dataset       (load-instances :csv (:train data/file-paths))
        f                   #(keyword (clj-ml-data/attr-name %))
        remove-attributes   (concat (map f (clj-ml-data/string-attributes train-dataset))
                                   (map f (clj-ml-data/nominal-attributes train-dataset)))
        numeric-train-data  (as-> train-dataset data
                                  (clj-ml-data/dataset-set-class data :SalePrice)
                                  (make-apply-filter :replace-missing-values {} data)
                                  (make-apply-filter :remove-attributes
                                                     {:attributes (into [:Id] remove-attributes)}
                                                     data))
        classifier (classifier-train (make-classifier :regression :linear) numeric-train-data)]
    (println (clj-ml-data/attribute-names numeric-train-data))
    (println classifier)
    {:classifier classifier
     :remove-attributes remove-attributes}))

(defn make-predictions
  [{:keys [classifier remove-attributes]}]
  (let [test-dataset       (load-instances :csv (:test data/file-paths))
        numeric-test-data  (as-> test-dataset data
                                 (make-apply-filter :replace-missing-values {} data)
                                 (make-apply-filter :remove-attributes
                                                    {:attributes (into [:Id] remove-attributes)}
                                                    data))]
    (println (clj-ml-data/attribute-names numeric-test-data))
    (println (first numeric-test-data))
    (classifier-predict-numeric classifier (first numeric-test-data))))

(defn reg
  [dataset]
  (classifier-train (make-classifier :regression :linear) dataset))

(clojure.set/difference #{:MSSubClass :LotArea :OverallQual :OverallCond :YearBuilt :YearRemodAdd :BsmtFinSF1 :BsmtFinSF2 :BsmtUnfSF :TotalBsmtSF :1stFlrSF :2ndFlrSF :LowQualFinSF :GrLivArea :BsmtFullBath :BsmtHalfBath :FullBath :HalfBath :BedroomAbvGr :KitchenAbvGr :TotRmsAbvGrd :Fireplaces :GarageCars :GarageArea :WoodDeckSF :OpenPorchSF :EnclosedPorch :3SsnPorch :ScreenPorch :PoolArea :MiscVal :MoSold :YrSold}
                        #{:MSSubClass :LotArea :OverallQual :OverallCond :YearBuilt :YearRemodAdd :1stFlrSF :2ndFlrSF :LowQualFinSF :GrLivArea :FullBath :HalfBath :BedroomAbvGr :KitchenAbvGr :TotRmsAbvGrd :Fireplaces :WoodDeckSF :OpenPorchSF :EnclosedPorch :3SsnPorch :ScreenPorch :PoolArea :MiscVal :MoSold :YrSold})

(:MSSubClass :LotArea :OverallQual :OverallCond :YearBuilt :YearRemodAdd :BsmtFinSF1 :BsmtFinSF2 :BsmtUnfSF :TotalBsmtSF :1stFlrSF :2ndFlrSF :LowQualFinSF :GrLivArea :BsmtFullBath :BsmtHalfBath :FullBath :HalfBath :BedroomAbvGr :KitchenAbvGr :TotRmsAbvGrd :Fireplaces :GarageCars :GarageArea :WoodDeckSF :OpenPorchSF :EnclosedPorch :3SsnPorch :ScreenPorch :PoolArea :MiscVal :MoSold :YrSold :SalePrice)
(:MSSubClass :LotArea :OverallQual :OverallCond :YearBuilt :YearRemodAdd :BsmtFinSF1 :BsmtFinSF2 :BsmtUnfSF :TotalBsmtSF :1stFlrSF :2ndFlrSF :LowQualFinSF :GrLivArea :BsmtFullBath :BsmtHalfBath :FullBath :HalfBath :BedroomAbvGr :KitchenAbvGr :TotRmsAbvGrd :Fireplaces :GarageCars :GarageArea :WoodDeckSF :OpenPorchSF :EnclosedPorch :3SsnPorch :ScreenPorch :PoolArea :MiscVal :MoSold :YrSold)
