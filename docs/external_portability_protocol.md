# Data1 Portability Protocol

Data1 is a local four-class task with fixed class order `[0,1,2,3]`. It does not share an assumed direct class mapping with the seven-class Tyndall task. The workflow trains local models from scratch to test pipeline or recipe portability; it is not deployment of a frozen seven-class classifier or direct seven-class Paper4 external validation. The documented split uses set 3 for training and sets 4–9 for testing. Majority ties resolve to the lowest class index. EV4 is complete with verdict `PASS_EV4_WITH_REQUIRED_CAVEATS`.
