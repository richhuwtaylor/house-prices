(ns house-prices.stats
  (:require [incanter.core :as i]
            [incanter.stats :as s]))

(defn jitter
  "Returns a jittering function that jitters a sample
  up to the given limit."
  [limit]
  (fn [x]
    (let [amount (- (rand (* 2 limit)) limit)]
      (+ x amount))))

(defn covariance
  "Calculates the covariance for xs and ys. This measures the
  tendency for the two variables to deviate from the mean in the
  same direction."
  [xs ys]
  (let [x-bar (s/mean xs)
        y-bar (s/mean xs)
        dx    (map (fn [x] (- x x-bar)) xs)
        dy    (map (fn [y] (- y y-bar)) ys)]
    (s/mean (map * dx dy))))

(defn variance
  "Calculates the variance (the mean square error) in xs."
  [xs]
  (let [x-bar        (s/mean xs)
        square-error (fn [x]
                       (i/pow (- x x-bar) 2))]
    (s/mean (map square-error xs))))

(defn standard-deviation
  "Calculates the standard deviation (square root of the variance)."
  [xs]
  (i/sqrt (variance xs)))

(defn pearson-correlation
  "Calculates the Pearson correlation for xs and ys."
  [xs ys]
  (/ (covariance xs ys)
     (* (standard-deviation xs)
        (standard-deviation ys))))

(defn predict
  "Calculate y-hats for the given coefficients and x matrices."
  [coefs x]
  (-> (i/trans coefs)
      (i/mmult x)))

(defn normal-equation
  "Calculate the coeffients of the ordinary least squares
  linear regression model."
  [x y]
  (let [xtx  (i/mmult (i/trans x) x)
        xtxi (i/solve xtx)
        xty  (i/mmult (i/trans x) y)]
    (i/mmult xtxi xty)))

(defn r-squared
  "Calculates the sum of squared residuals divided by the sum of
  squared differences from the mean."
  [coefs x y]
  (let [fitted      (i/mmult x coefs)
        residuals   (i/minus y fitted)
        differences (i/minus y (s/mean y))
        rss         (i/sum-of-squares residuals)
        ess         (i/sum-of-squares differences)]
    (- 1 (/ rss ess))))

(defn adj-r-squared
  "Calculates the R-bar squared. This depends on the number of sample size n
  and the number of parameters p and will only increase if the addition of a
  new independent variable increases R-squared more than would be expected
  due to chance."
  [coefs x y]
  (let [r-squared (r-squared coefs x y)
        n         (count y)
        p         (count coefs)]
    (- 1
       (* (- 1 r-squared)
          (/ (dec n)
             (dec (- n p)))))))


