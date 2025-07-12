# Heart Rate Prediction Model (Work in progess)

This is just a personal fun project to see how accurately I can preddict unpredictable - my heart rate. Using 8 months of my heart rate data from Whoop, I am trying to capture at least some parts of the daily, work day and montly seasonality. Lets see how it goes. 

## Input data format
Daily heart rate records. My dataset is currently sample every 6 seconds.

```json
[
    {
        "timestamp": 1736035200223,
        "datetime": "2025-01-05T01:00:00.223Z",
        "heart_rate": 89
    },
    ...
]
```

## Results
**TLDR: My daily commute and excercise routine is not consistent to establish any daily or workweek trent. As results, model predicts HR in range 47.9 to 96.5 BPM, while dataset ranges 44.8 to 185.1 BPM.** There might be a better ways to capture activity and include them in the modeling as separate events, but well, that is work for later.