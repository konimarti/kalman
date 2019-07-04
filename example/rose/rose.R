df<-read.csv(file="p133_out.csv", header=T)

png("rose.png")

plot(df$m, ylab="y(k)", xlab="Time",type="l",lwd=2,lty=2,col="green")
lines(df$f,col="red",lwd=2)
legend("topright", legend=c("Input signal","Filtered signal"), lty=c(2,1), col=c("green","red"))

dev.off()
