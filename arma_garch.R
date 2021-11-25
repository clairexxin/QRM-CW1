require(rugarch)
data <- read.csv("QRM-2021-cw1-data-a.csv")
price <- data$SX5E
n <- length(price)
r <- log(price[2:n]/price[1:n-1]) * 100

spec <- ugarchspec(variance.model = list(model = "sGARCH", 
                                 garchOrder = c(1, 1)), 
                   mean.model     = list(armaOrder = c(1, 1)),
                   distribution.model = "std")
garch <- ugarchfit(spec = spec, data = r)
garch
infocriteria(garch)
sigma <- sigma(garch)
resid <- residuals(garch, standardize=TRUE)
write.csv(sigma,row.names = FALSE, "arma-garch-sigma.csv")
write.csv(resid,row.names = FALSE, "arma-garch-resid.csv")