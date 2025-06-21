__kernel void compute(__global const int* changes, __global int* account,
                      __global int* sum, int clients, int periods) {
  // TODO insert your code here
  int i = get_global_id(0);  // client index

  if (i >= clients) return;

  // copy first period as is
  account[i] = changes[i];
  atomic_add(&sum[0], account[i]);

  for (int j = 1; j < periods; j++) {
    int idx = j * clients + i;
    int prev_idx = (j - 1) * clients + i;
    account[idx] = account[prev_idx] + changes[idx];

    // update sum for the period
    atomic_add(&sum[j], account[idx]);
  }
}
