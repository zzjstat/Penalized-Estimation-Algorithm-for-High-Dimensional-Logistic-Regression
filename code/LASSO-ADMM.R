
# lasso的ADMM算法
# 逻辑回归的负对数似然损失函数
logistic_nll <- function(beta, X, y) {
  linear_pred <- X %*% beta  # 线性预测
  prob <- 1 / (1 + exp(-linear_pred))  # 计算sigmoid概率
  nll <- -sum(y * log(prob + 1e-10) + (1 - y) * log(1 - prob + 1e-10))
  return(1/nrow(X)*nll)
}

# 定义损失函数，包括正则化项
loss_function <- function(beta, z, X, y, lambda, theta, eta) {
  nll <- logistic_nll(beta, X, y)
  l1_penalty <- lambda * sum(abs(z))  # L1 正则化项
  linear_term <- -sum(theta * (z - beta))  # 线性项
  l2_penalty <- (eta / 2) * sum((z - beta)^2)  # L2 正则化项
  total_loss <- nll + l1_penalty + linear_term + l2_penalty
  return(total_loss)
}

# 计算梯度
gradient_function <- function(beta, z, X, y, lambda, theta, eta) {
  # 负对数似然的梯度
  prob <- 1 / (1 + exp(-X %*% beta))  
  grad_nll <- 1/nrow(X)*t(X) %*% (prob - y) 
  
  # 线性项的梯度
  linear_grad <- theta  
  
  # L2 正则化的梯度
  l2_grad <- eta * (beta - z)  
  
  # 总体梯度
  total_gradient <- grad_nll  + linear_grad + l2_grad
  return(total_gradient)
}

# ADMM 算法实现
admm_logistic_lasso <- function(X, y, lambda, eta = 1, 
                                max_iter = 5000, tol = 1e-6) {
  n <- nrow(X)
  p <- ncol(X)

  # 初始化 beta
  beta <- tryCatch(
    { coef(glm(y~X,family = "binomial"))[-1]},
    warning = function(w){return(rep(0,p))},
    error = function(e) { return(rep(0,p)) }
  )
  # 初始化 z
  z <-tryCatch(
    { coef(glm(y~X,family = "binomial"))[-1]},
    warning = function(w){return(rep(0,p))},
    error = function(e) { return(rep(0,p)) }
  )
  z[which(abs(z)<1)]=0
  theta <- rep(1, p)     # 初始化 theta
  
  for (iter in 1:max_iter) {
    # 更新 beta
    beta_optim <- optim(beta, loss_function, z = z, X = X, y = y, 
                        lambda = lambda, theta = theta, eta = eta, 
                        method = "BFGS", gr = gradient_function)
    beta <- beta_optim$par
    
    # 更新 z，通过软阈值
    z_old <- z
    z <- pmax(0, abs(beta + theta) - lambda/eta)*sign(beta + theta)
    
    # 更新 theta
    theta <- theta + beta - z
    beta[which(abs(beta)<lambda)]=0
    # 检查收敛
    if (sum(abs(z - z_old)) < tol) {
      break
    }
  }
  
  return(beta)
}
