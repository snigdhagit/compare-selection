import numpy as np
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri

def plotRisk(df_risk):
    robjects.r("""
               library("ggplot2")
               library("magrittr")
               library("tidyr")
               library("dplyr")

               plot_risk <- function(df_risk, outpath="plots/", resolution=300, height= 7.5, width=15)
                { 
                   date = 1:length(unique(df_risk$snr))
                   df_risk = filter(df_risk, metric == "Full")
                   df = cbind(df_risk, date)
                   risk = df %>%
                   gather(key, value, sel.MLE, rand.LASSO, LASSO) %>%
                   ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                   geom_point(size=3) +
                   geom_line(aes(linetype=key), size=1) +
                   ylim(0.01,1.2)+
                   labs(y="relative risk", x = "Signal regimes: snr") +
                   scale_x_continuous(breaks=1:length(unique(df_risk$snr)), label = sapply(df_risk$snr, toString)) +
                   theme(legend.position="top", legend.title = element_blank())
                   indices = sort(c("sel.MLE", "rand.LASSO", "LASSO"), index.return= TRUE)$ix
                   names = c("sel-MLE", "rand-LASSO", "LASSO")
                   risk = risk + scale_color_manual(labels = names[indices], values=c("#008B8B", "#104E8B","#B22222")[indices]) +
                   scale_shape_manual(labels = names[indices], values=c(15, 17, 16)[indices]) +
                                      scale_linetype_manual(labels = names[indices], values = c(1,1,2)[indices])
                                      outfile = paste(outpath, 'risk.png', sep="")
                   ggsave(outfile, plot = risk, dpi=resolution, dev='png', height=height, width=width, units="cm")}
                """)

    robjects.pandas2ri.activate()
    r_df_risk = robjects.conversion.py2ri(df_risk)
    R_plot = robjects.globalenv['plot_risk']
    R_plot(r_df_risk)

def plotCoveragePower(df_inference):
    robjects.r("""
               library("ggplot2")
               library("magrittr")
               library("tidyr")
               library("reshape")
               library("cowplot")
               library("dplyr")
               
               plot_coverage_lengths <- function(df_inference, outpath="plots/", 
                                                 resolution=200, height_plot1= 6.5, width_plot1=12, 
                                                 height_plot2=13, width_plot2=13)
               {
                 snr.len = length(unique(df_inference$snr))
                 df_inference = arrange(df_inference, method)
                 target = toString(df_inference$target[1])
                 df = data.frame(snr = sapply(unique(df_inference$snr), toString),
                                 MLE = 100*df_inference$coverage[((2*snr.len)+1):(3*snr.len)],
                                 Lee = 100*df_inference$coverage[1:snr.len],
                                 Naive = 100*df_inference$coverage[((3*snr.len)+1):(4*snr.len)])
                 if(target== "selected"){
                      data.m <- melt(df, id.vars='snr')
                      coverage = ggplot(data.m, aes(snr, value)) + 
                                 geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                                 geom_hline(yintercept = 90, linetype="dotted") +
                                 labs(y="coverage: partial", x = "Signal regimes: snr") +
                                 theme(legend.position="top", 
                                       legend.title = element_blank()) 
                      coverage = coverage + 
                                 scale_fill_manual(labels = c("MLE-based","Lee", "Naive"), values=c("#008B8B", "#B22222", "#FF6347"))} else{
                 df = cbind(df, Liu = 100*df_inference$coverage[((snr.len)+1):(2*snr.len)])
                 df <- df[c("snr", "MLE", "Liu", "Lee", "Naive")]
                 data.m <- melt(df, id.vars='snr')
                 coverage = ggplot(data.m, aes(snr, value)) + 
                            geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                            geom_hline(yintercept = 90, linetype="dotted") +
                            labs(y="coverage: full", x = "Signal regimes: snr") +
                            theme(legend.position="top", legend.title = element_blank()) 
                  coverage = coverage + 
                  scale_fill_manual(labels = c("MLE-based", "Liu", "Lee", "Naive"), values=c("#008B8B", "#104E8B", "#B22222", "#FF6347"))}
  
                 outfile = paste(outpath, 'coverage.png', sep="")
                 ggsave(outfile, plot = coverage, dpi=resolution, dev='png', height=height_plot1, width=width_plot1, units="cm")
               
                 df = data.frame(snr = sapply(unique(df_inference$snr), toString),
                                 MLE = 100*df_inference$sel.power[((2*snr.len)+1):(3*snr.len)],
                                 Lee = 100*df_inference$sel.power[1:snr.len])
                 if(target== "selected"){
                   data.m <- melt(df, id.vars='snr')
                   sel_power = ggplot(data.m, aes(snr, value)) + 
                               geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                               labs(y="power: partial", x = "Signal regimes: snr") +
                               theme(legend.position="top", legend.title = element_blank()) 
                   sel_power = sel_power + scale_fill_manual(labels = c("MLE-based","Lee"), values=c("#008B8B", "#B22222"))} else{
                   df = cbind(df, Liu = 100*df_inference$sel.power[((snr.len)+1):(2*snr.len)])
                   df <- df[,c("snr", "MLE", "Liu", "Lee")]
                   data.m <- melt(df, id.vars='snr')
                   sel_power = ggplot(data.m, aes(snr, value)) + 
                               geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                               labs(y="power: full", x = "Signal regimes: snr") +
                               theme(legend.position="top", legend.title = element_blank()) 
                   sel_power = sel_power + scale_fill_manual(labels = c("MLE-based","Liu","Lee"), values=c("#008B8B", "#104E8B", "#B22222"))}
  
                 outfile = paste(outpath, 'selective_power.png', sep="")
                 ggsave(outfile, plot = sel_power, dpi=resolution, dev='png', height=height_plot1, width=width_plot1, units="cm")
  
               if(target== "selected"){
                   test_data <-data.frame(MLE = filter(df_inference, method == "MLE")$length,
                   Lee = filter(df_inference, method == "Lee")$length,
                   Naive = filter(df_inference, method == "Naive")$length,
                   date = 1:length(unique(df_inference$snr)))
                   lengths = test_data %>%
                             gather(key, value, MLE, Lee, Naive) %>%
                             ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                             geom_point(size=3) +
                             geom_line(aes(linetype=key), size=1) +
                             ylim(0.,max(test_data$MLE, test_data$Lee, test_data$Naive) + 0.2)+
                             labs(y="lengths:partial", x = "Signal regimes: snr") +
                             scale_x_continuous(breaks=1:length(unique(df_inference$snr)), label = sapply(unique(df_inference$snr), toString))+
                             theme(legend.position="top", legend.title = element_blank())
    
                   indices = sort(c("MLE", "Lee", "Naive"), index.return= TRUE)$ix
                   names = c("MLE-based", "Lee", "Naive")
                   lengths = lengths + scale_color_manual(labels = names[indices], values=c("#008B8B","#B22222", "#FF6347")[indices]) +
                             scale_shape_manual(labels = names[indices], values=c(15, 17, 16)[indices]) +
                             scale_linetype_manual(labels = names[indices], values = c(1,1,2)[indices])} else{
                   test_data <-data.frame(MLE = filter(df_inference, method == "MLE")$length,
                                          Lee = filter(df_inference, method == "Lee")$length,
                                          Naive = filter(df_inference, method == "Naive")$length,
                                          Liu = filter(df_inference, method == "Liu")$length,
                                          date = 1:length(unique(df_inference$snr)))
                   lengths= test_data %>%
                            gather(key, value, MLE, Lee, Naive, Liu) %>%
                            ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                            geom_point(size=3) +
                            geom_line(aes(linetype=key), size=1) +
                            ylim(0.,max(test_data$MLE, test_data$Lee, test_data$Naive, test_data$Liu) + 0.2)+
                            labs(y="lengths: full", x = "Signal regimes: snr") +
                            scale_x_continuous(breaks=1:length(unique(df_inference$snr)), label = sapply(unique(df_inference$snr), toString))+
                            theme(legend.position="top", legend.title = element_blank())
         
                   indices = sort(c("MLE", "Liu", "Lee", "Naive"), index.return= TRUE)$ix
                   names = c("MLE-based", "Lee", "Naive", "Liu")
                   lengths = lengths + scale_color_manual(labels = names[indices], values=c("#008B8B","#B22222", "#FF6347", "#104E8B")[indices]) +
                             scale_shape_manual(labels = names[indices], values=c(15, 17, 16, 15)[indices]) +
                             scale_linetype_manual(labels = names[indices], values = c(1,1,2,1)[indices])}
  
               prop = filter(df_inference, method == "Lee")$prop.infty
               df = data.frame(snr = sapply(unique(df_inference$snr), toString),
               infinite = 100*prop)
               data.prop <- melt(df, id.vars='snr')
               pL = ggplot(data.prop, aes(snr, value)) +
                    geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                    labs(y="infinite intervals (%)", x = "Signal regimes: snr") +
                    theme(legend.position="top", 
                    legend.title = element_blank()) 
               pL = pL + scale_fill_manual(labels = c("Lee"), values=c("#B22222"))
               prow <- plot_grid( pL + theme(legend.position="none"),
                                  lengths + theme(legend.position="none"),
                                  align = 'vh',
                                  hjust = -1,
                                  ncol = 1)
  
               legend <- get_legend(lengths+ theme(legend.direction = "horizontal",legend.justification="center" ,legend.box.just = "bottom"))
               p <- plot_grid(prow, ncol=1, legend, rel_heights = c(2., .2)) 
               outfile = paste(outpath, 'length.png', sep="")
               ggsave(outfile, plot = p, dpi=resolution, dev='png', height=height_plot2, width=width_plot2, units="cm")}
               """)

    robjects.pandas2ri.activate()
    r_df_inference = robjects.conversion.py2ri(df_inference)
    R_plot = robjects.globalenv['plot_coverage_lengths']
    R_plot(r_df_inference)
