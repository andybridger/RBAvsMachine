# =============================================================================
# Machine Learning Model Wrappers
# =============================================================================

fit_historical_mean <- function(y_train, X_train = NULL, X_test = NULL) {
  list(forecast = mean(y_train, na.rm = TRUE), model = "Historical Mean")
}

fit_ar1 <- function(y_train, X_train = NULL, X_test = NULL) {
  if (length(y_train) < 5) return(list(forecast = mean(y_train, na.rm = TRUE), model = "AR(1)"))
  mod <- tryCatch(
    ar(y_train, order.max = 1, aic = FALSE, method = "ols"),
    error = function(e) NULL
  )
  if (is.null(mod) || length(mod$ar) == 0) {
    return(list(forecast = mean(y_train, na.rm = TRUE), model = "AR(1)"))
  }
  fc <- predict(mod, n.ahead = 1)$pred[1]
  list(forecast = fc, model = "AR(1)")
}

fit_ridge <- function(y_train, X_train, X_test) {
  if (ncol(X_train) == 0 || nrow(X_train) < 10) {
    return(list(forecast = mean(y_train, na.rm = TRUE), model = "Ridge"))
  }
  cv_fit <- tryCatch(
    glmnet::cv.glmnet(X_train, y_train, alpha = 0, nfolds = min(10, nrow(X_train))),
    error = function(e) NULL
  )
  if (is.null(cv_fit)) return(list(forecast = mean(y_train, na.rm = TRUE), model = "Ridge"))
  fc <- predict(cv_fit, newx = X_test, s = "lambda.min")[1, 1]
  list(forecast = fc, model = "Ridge")
}

fit_lasso <- function(y_train, X_train, X_test) {
  if (ncol(X_train) == 0 || nrow(X_train) < 10) {
    return(list(forecast = mean(y_train, na.rm = TRUE), model = "LASSO"))
  }
  cv_fit <- tryCatch(
    glmnet::cv.glmnet(X_train, y_train, alpha = 1, nfolds = min(10, nrow(X_train))),
    error = function(e) NULL
  )
  if (is.null(cv_fit)) return(list(forecast = mean(y_train, na.rm = TRUE), model = "LASSO"))
  fc <- predict(cv_fit, newx = X_test, s = "lambda.min")[1, 1]
  list(forecast = fc, model = "LASSO")
}

fit_elastic_net <- function(y_train, X_train, X_test) {
  if (ncol(X_train) == 0 || nrow(X_train) < 10) {
    return(list(forecast = mean(y_train, na.rm = TRUE), model = "Elastic Net"))
  }
  cv_fit <- tryCatch(
    glmnet::cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = min(10, nrow(X_train))),
    error = function(e) NULL
  )
  if (is.null(cv_fit)) return(list(forecast = mean(y_train, na.rm = TRUE), model = "Elastic Net"))
  fc <- predict(cv_fit, newx = X_test, s = "lambda.min")[1, 1]
  list(forecast = fc, model = "Elastic Net")
}

fit_random_forest <- function(y_train, X_train, X_test) {
  if (ncol(X_train) == 0 || nrow(X_train) < 10) {
    return(list(forecast = mean(y_train, na.rm = TRUE), model = "Random Forest"))
  }
  train_df <- data.frame(y = y_train, X_train)
  test_df <- data.frame(X_test)
  colnames(test_df) <- colnames(train_df)[-1]
  rf_fit <- tryCatch(
    randomForest::randomForest(y ~ ., data = train_df, ntree = 500, importance = FALSE),
    error = function(e) NULL
  )
  if (is.null(rf_fit)) return(list(forecast = mean(y_train, na.rm = TRUE), model = "Random Forest"))
  fc <- predict(rf_fit, newdata = test_df)[1]
  list(forecast = fc, model = "Random Forest")
}

fit_xgboost <- function(y_train, X_train, X_test) {
  if (ncol(X_train) == 0 || nrow(X_train) < 10) {
    return(list(forecast = mean(y_train, na.rm = TRUE), model = "XGBoost"))
  }
  dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgboost::xgb.DMatrix(data = X_test)
  params <- list(
    objective = "reg:squarederror",
    max_depth = 3,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  xgb_fit <- tryCatch(
    xgboost::xgb.train(
      params = params, data = dtrain,
      nrounds = 100, verbose = 0,
      early_stopping_rounds = 10,
      watchlist = list(train = dtrain)
    ),
    error = function(e) NULL
  )
  if (is.null(xgb_fit)) return(list(forecast = mean(y_train, na.rm = TRUE), model = "XGBoost"))
  fc <- predict(xgb_fit, newdata = dtest)[1]
  list(forecast = fc, model = "XGBoost")
}

get_model_functions <- function() {
  list(
    "Historical Mean" = fit_historical_mean,
    "AR(1)"           = fit_ar1,
    "Ridge"           = fit_ridge,
    "LASSO"           = fit_lasso,
    "Elastic Net"     = fit_elastic_net,
    "Random Forest"   = fit_random_forest,
    "XGBoost"         = fit_xgboost
  )
}
