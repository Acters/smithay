#!/usr/bin/env bash
set -euo pipefail

TEST_BIN="${1:-./vk_export_test}"
LOG="${2:-prime_enomem_trace.log}"

if [[ ! -x "$TEST_BIN" ]]; then
    echo "Test binary is not executable: $TEST_BIN" >&2
    exit 1
fi

if ! command -v bpftrace >/dev/null 2>&1; then
    echo "bpftrace is required." >&2
    echo "On Arch Linux: sudo pacman -S bpftrace" >&2
    exit 1
fi

sudo -v

TRACEFS=/sys/kernel/tracing
[[ -r "$TRACEFS/available_filter_functions" ]] ||
    TRACEFS=/sys/kernel/debug/tracing

AVAILABLE="$TRACEFS/available_filter_functions"
if [[ ! -r "$AVAILABLE" ]]; then
    echo "Cannot read available_filter_functions under tracefs." >&2
    exit 1
fi

candidates=(
    __nv_drm_gem_nvkms_memory_prime_get_sg_table
    nv_drm_gem_prime_get_sg_table
    drm_gem_map_dma_buf
    dma_buf_map_attachment
    i915_gem_object_get_pages_dmabuf
    i915_gem_object_pin_pages
    i915_vma_pin_ww
    i915_gem_execbuffer2_ioctl
)

tmp_bt="$(mktemp --suffix=.bt)"
cleanup() {
    rm -f "$tmp_bt"
}
trap cleanup EXIT

cat >"$tmp_bt" <<'EOF'
BEGIN
{
    printf("Tracing negative/error-pointer returns in the DMA-BUF PRIME path...\n");
}
EOF

found=0
for fn in "${candidates[@]}"; do
    if grep -Eq "(^|[[:space:]])${fn}([[:space:]]|$)" "$AVAILABLE"; then
        cat >>"$tmp_bt" <<EOF

kretprobe:${fn}
/(int64)retval < 0/
{
    printf("%-62s ret=%d pid=%d comm=%s\n",
           probe, (int64)retval, pid, comm);
    print(kstack(12));
}
EOF
        printf 'Will trace: %s\n' "$fn"
        found=$((found + 1))
    else
        printf 'Unavailable: %s\n' "$fn"
    fi
done

if (( found == 0 )); then
    echo "None of the useful symbols are kprobe-visible on this kernel." >&2
    exit 1
fi

: >"$LOG"
sudo bpftrace "$tmp_bt" >"$LOG" 2>&1 &
tracer_pid=$!

stop_tracer() {
    sudo kill -INT "$tracer_pid" 2>/dev/null || true
    wait "$tracer_pid" 2>/dev/null || true
}
trap 'stop_tracer; cleanup' EXIT

# Give bpftrace time to attach all probes.
sleep 2

set +e
"$TEST_BIN"
test_status=$?
set -e

sleep 1
stop_tracer
trap cleanup EXIT

echo
echo "Test exit status: $test_status"
echo "Trace written to: $LOG"
echo
cat "$LOG"
