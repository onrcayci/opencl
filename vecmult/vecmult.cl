kernel void vecmult(global float4 *vec1, global float4 *vec2,
                    global float4 *result) {
  int i = get_global_id(0);
  result[i] = vec1[i] * vec2[i];
}