## Overall
- I want to build a trading bot which can analyze stock in minutes, and do actual trading.
- You as the role of AI coding assistant, you are encouraged to debate and discuss about any of my ideas, provide your thoughts, e.g `Future` or `Experiment`, and provide your own ideas, suggestions, and feedback.
- Annotations: 
  - `Future` means in future, I will add that feature, but not now, so aware of project architecture and design for ease of maintenance and scalability.
  - `Experiment` means I consider to make experiment.
- Target stock: technology and AI chain stocks
  - Some reference in: `docs/architecture/research/AIChain.md`
    + e.g. TSM ADR, NVDIA, ASML, AMD, AVGO, MRVL, QCOM, INTC, ARM, SNPS, CDNS, SIEGY, AMAT, LRCX, KLAC, NYSE: ASX, AMKR, MU, ANET, COHR, LITE, FN, DELL, SMCI, HPE... I think around 50 stocks is a good start.
    + `Experiment`: 
      + First I think we just put all of that normally as v1.
      + should we record data from 2021 to now, because I think around it when AI hype started.
      + Second, after workflow done with first step, how about add category for each stock, e.g. semiconductor, networking, cloud, etc. 
  - `Future`: other sectors, e.g. healthcare, energy, etc.
  - Data source:
    + I think we can use Alpaca API for stock data first and future is trading execution.
    + record each minute
- Algorithm: 
  - ML like Random Forest and XGBoost may be a good choice
  - Carefully process the data, and plan the best algorithm to train the model.

- For first version, I only need to train model on stock and algorithm to trade and see if model can find patterns and make profit. Aware of fee trading (e.g. on binance stock trading). Design appropriate backtesting and simulation to avoid loss. Because data in minute, so model can trade on minute basis.
  + Begin with 2000 USD from January 2026 to now, how much it will earn. I want you optimize the model to make profit, and also avoid loss. 
  + I wonder should we train model after each month to see if it can appropve model performance. Then if yes then try with each week.
- Let write docs/architecture/workflow.md which describe the  architecture in modular monlithic architecture, the file must include 2 main part, first is the folder architecture which include folder and main files, second is the workflow, which also present which folder or main files relate to that step