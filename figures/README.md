# Anomaly detection in quasi-periodic energy consumption data series: a comparison of algorithms

In this folder you will find full-size versions of the figures in the article:

<p float="center">
<img src="fig1.png" alt="Fig1" style="height: 500px; width:650px;"/>
<br><b>Fig1.</b> Example of the fridge energy consumption data series. The time series is formed by subsequent ON-OFF cycles and is quasi-periodical.
</p>
<br>&nbsp;

<p float="center">
  <img src="fig2_left.png" alt="Fig1" style="height: 350px; width:500px;"/>
  <img src="fig2_right.png" alt="Fig1" style="height: 350px; width:500px;"/>
  <br><b>Fig2.</b> The power spectrum computed by the periodicity pre-processor (right) on the fridge energy consumption time series (left). The period detected for an ON-OFF cycle is about 80 minutes for the analyzed data set.
</p>
<br>&nbsp;

<p float="center">
<img src="fig3.png" alt="Fig3" style="height: 300px; width:600px;"/>
  <br><b>Fig3.</b> The interface of the GT anomaly annotator at work on the fridge time series. The user can specify the anomalies and add  meta-data to them.  The user has annotated the currently selected GT anomaly, shown in red, with the <i>Continuous ON state</i> label.
</p>
<br>&nbsp;

<p float="center">
<img src="fig4_fridge1.png" alt="Fig4" style="height: 270px; width:300px;"/>
<img src="fig4_fridge2.png" alt="Fig4" style="height: 270px; width:300px;"/>
<img src="fig4_fridge3.png" alt="Fig4" style="height: 270px; width:300px;"/>
  <br><b>Fig4.</b> The anomaly type distribution on the three fridge energy consumption data series.
</p>
<br>&nbsp;

<p float="center">
<img src="fig5_fridge1.png" alt="Fig5" style="height: 240px; width:300px;"/>
<img src="fig5_fridge2.png" alt="Fig5" style="height: 240px; width:300px;"/>
<img src="fig5_fridge3.png" alt="Fig5" style="height: 240px; width:300px;"/>
  <br><b>Fig5.</b> The anomaly duration distribution on the fridge energy consumption data sets.  The distributions of Fridge1 and Fridge2 are centered close the time series period, which suggests the presence of anomalies shorter than an ON-OFF cycle whereas the distribution of Fridge3 is centered around values higher than the mean ON-OFF cycle duration.
</p>
<br>&nbsp;

<p float="center">
<img src="fig6.png" alt="Fig6" style="height: 420px; width:600px;"/>
  <br><b>Fig6.</b> Comparison of the  performances of all the algorithms on all the  appliances and across all the training duration periods and window sizes. The methods are ordered in descending order of the median values of the F1 score.  
</p>
<br>&nbsp;

<p float="center">
<img src="fig7_fridge1.png" alt="Fig7" style="height: 210px; width:300px;"/>
<img src="fig7_fridge2.png" alt="Fig7" style="height: 210px; width:300px;"/>
<img src="fig7_fridge3.png" alt="Fig7" style="height: 210px; width:300px;"/>
  <br><b>Fig7.</b> Break down of the performance of all the algorithms by appliance. The methods are ordered by descending median value of the F1 score. 
</p>
<br>&nbsp;

<p float="center">
<img src="Q2/fig8_fridge3_ISOF.png" alt="Fig8" style="height: 200px; width:200px;"/>
<img src="Q2/fig8_fridge3_OC_SVM.png" alt="Fig8" style="height: 200px; width:200px;"/>
<img src="Q2/fig8_fridge3_LSTM.png" alt="Fig8" style="height: 200px; width:200px;"/>
  <img src="Q2/fig8_fridge3_LOF.png" alt="Fig8" style="height: 200px; width:200px;"/>
</p>
<p float="center">
<img src="Q2/fig8_fridge3_GRU.png" alt="Fig8" style="height: 200px; width:200px;"/>
<img src="Q2/fig8_fridge3_GRU-AE.png" alt="Fig8" style="height: 200px; width:200px;"/>
<img src="Q2/fig8_fridge3_LSTM-AE.png" alt="Fig8" style="height: 200px; width:200px;"/>
<img src="Q2/fig8_fridge3_GRU-MS.png" alt="Fig8" style="height: 200px; width:200px;"/>
</p>
<p float="center">
<img src="Q2/fig8_fridge3_LSTM-MS.png" alt="Fig8" style="height: 200px; width:200px;"/>
<img src="Q2/fig8_fridge3_basic_statistics.png" alt="Fig8" style="height: 200px; width:200px;"/>
  <br><b>Fig8.</b> Variation of the F1 score with the duration of the training sub-sequence. The AR and ARIMA method did not complete the training with all the periods.
</p>
<br>&nbsp;

<p float="center">
<img src="Q3/fig9_fridge1_ISOF.png" alt="Fig9" style="height: 200px; width:200px;"/>
<img src="Q3/fig9_fridge1_OC_SVM.png" alt="Fig9" style="height: 200px; width:200px;"/>
<img src="Q3/fig9_fridge1_LSTM.png" alt="Fig9" style="height: 200px; width:200px;"/>
<img src="Q3/fig9_fridge1_LOF.png" alt="Fig9" style="height: 200px; width:200px;"/>
</p>
<p float="center">
<img src="Q3/fig9_fridge1_GRU.png" alt="Fig9" style="height: 200px; width:200px;"/>
<img src="Q3/fig9_fridge1_GRU-AE.png" alt="Fig9" style="height: 200px; width:200px;"/>
<img src="Q3/fig9_fridge1_LSTM-AE.png" alt="Fig9" style="height: 200px; width:200px;"/>
<img src="Q3/fig9_fridge1_GRU-MS.png" alt="Fig9" style="height: 200px; width:200px;"/>
<p float="center">
<img src="Q3/fig9_fridge1_LSTM-MS.png" alt="Fig9" style="height: 200px; width:200px;"/>
  <br><b>Fig9.</b> Variation of the  F1 score with the size (in periods) of the sliding window. The AR and ARIMA method did not complete the training with all the periods.
</p>
<br>&nbsp;
<p float="center">
<img src="Q4/fig10_fridge1_on_fridge2.png" alt="Fig10" style="height: 200px; width:500px;"/>
<img src="Q4/fig10_fridge1_on_fridge3.png" alt="Fig10" style="height: 200px; width:500px;"/>
</p>
<p float="center">
<img src="Q4/fig10_fridge2_on_fridge1.png" alt="Fig10" style="height: 200px; width:500px;"/>
<img src="Q4/fig10_fridge2_on_fridge3.png" alt="Fig10" style="height: 200px; width:500px;"/>
<p float="center">
<img src="Q4/fig10_fridge3_on_fridge1.png" alt="Fig10" style="height: 200px; width:500px;"/>
<img src="Q4/fig10_fridge3_on_fridge2.png" alt="Fig10" style="height: 200px; width:500px;"/>
<br><b>Fig10.</b> Comparison of the generalization performance of the top-5 methods. The orange bar represents the baseline F1 value (i.e., training and testing done on the same dataset), the blue bar denotes the F1 value achieved by fine tuning the threshold on the validation set of the target appliance, and the green bar shows the performances obtained using the trained algorithm without fine tuning.
</p>
<br>&nbsp;


