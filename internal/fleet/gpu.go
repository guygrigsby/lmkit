package fleet

import (
	"fmt"
	"strconv"
	"strings"
)

// gpuVendorEnv maps a manifest gpu vendor to its device-visibility env var.
var gpuVendorEnv = map[string]string{
	"rocm": "HIP_VISIBLE_DEVICES",
	"cuda": "CUDA_VISIBLE_DEVICES",
}

// GPUEnv maps a manifest gpu spec ("<vendor>:<index>") to an environment
// assignment string, e.g. "rocm:0" -> "HIP_VISIBLE_DEVICES=0" and
// "cuda:1" -> "CUDA_VISIBLE_DEVICES=1". It errors on a missing colon, an
// unknown vendor, or a non-numeric / negative index.
func GPUEnv(gpu string) (string, error) {
	vendor, index, ok := strings.Cut(gpu, ":")
	if !ok {
		return "", fmt.Errorf("gpu %q: missing colon, want <vendor>:<index>", gpu)
	}
	if strings.Contains(index, ":") {
		return "", fmt.Errorf("gpu %q: too many parts, want <vendor>:<index>", gpu)
	}
	envName, known := gpuVendorEnv[vendor]
	if !known {
		return "", fmt.Errorf("gpu %q: unknown vendor %q (want rocm or cuda)", gpu, vendor)
	}
	n, err := strconv.Atoi(index)
	if err != nil || n < 0 {
		return "", fmt.Errorf("gpu %q: index must be a non-negative integer", gpu)
	}
	return fmt.Sprintf("%s=%d", envName, n), nil
}
