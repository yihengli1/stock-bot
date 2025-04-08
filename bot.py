from ib_insync import IB, util

# Create an IB instance
ib = IB()

# Connect to TWS or IB Gateway
# Replace '127.0.0.1' with the host, and adjust port and clientId if needed.
ib.connect('127.0.0.1', 7496, clientId=1)

print("Connected:", ib.isConnected())

# Fetch account summary as an example
account_summary = ib.accountSummary()
for item in account_summary:
    print(item)

# Disconnect from the API
ib.disconnect()
