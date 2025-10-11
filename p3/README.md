# Project 3: GPU-Accelerated Savings Account System

**Course:** High Performance Computing, University of Vienna (2025S)

A high-performance OpenCL implementation for computing cumulative savings account balances across multiple time periods, optimized for NVIDIA Tesla K20m GPUs.

## Overview

This project implements a GPU-accelerated system for processing savings account transactions. Given a 2D array of account changes over time, the system computes cumulative balances for each customer across all periods and calculates total money in the system per period.

### Problem Description

**Input:**
- 2D array `changes[PERIODS][CLIENTS]`: Money added to each customer's account per period
  - Each column represents one customer's transaction history
  - Each row represents all transactions in one time period

**Output:**
- 2D array `account[PERIODS][CLIENTS]`: Cumulative balance for each customer per period
- 1D array `sum[PERIODS]`: Total money across all accounts per period

**Example:**
```
Customer:     0    1    2
Period 0:   +10  +20  +15  � Account: [10, 20, 15], Sum: 45
Period 1:    +5  +10   +8  � Account: [15, 30, 23], Sum: 68
Period 2:    +3   +7   +2  � Account: [18, 37, 25], Sum: 80
```
