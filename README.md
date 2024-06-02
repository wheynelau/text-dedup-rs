<center><img src="https://camo.githubusercontent.com/adc94e53c011a5ee3606dad1223c776c169d32e26055a6d9a01ef28fb0a55964/68747470733a2f2f70617065722d6174746163686d656e74732e64726f70626f782e636f6d2f735f353445314239364546464546443239343536323930324443354239393731443335434436423635304243383744313230303341333041343635313737363230315f313538363531353635343537335f737469636b65722e77656270"/ style="background-color:white;"></center>

# MAKING DEDUP GO BRRRRRR 🚀🚀🚀 (... somewhat)

> "The First Rule of Program Optimization: Don't do it. The Second Rule of Program Optimization (for experts only!): Don't do it yet." — Michael A. Jackson
> 
> "If it ain't broke, don't fix it." — Bert Lance


## Pyspark

The issues I got from pyspark was that it mostly handled the data in a distributed manner. This may be of use for massive datasets, but the use case here it was difficult to improve the performance.

Using the code in this branch, the dedup for `benchmark_news.py` process took about 60s for the rust implementation and 16s on pyspark. 

