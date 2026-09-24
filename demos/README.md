# XGBoost.Managed demos

The C# ports of the Python demos (`demo`), run against the fully managed c# port.

```shell
cd csharpport/demos/XGBoost.Managed.Demos
dotnet run -c Release                 # list the demos
dotnet run -c Release -- <demo>       # run one demo
dotnet run -c Release -- all          # run every demo
```

No native `xgboost` library is needed.