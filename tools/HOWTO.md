# Disable the Serial Console on ttyAMA0

The Pi ships with a kernel console on the same UART the Rosman5 board uses. Remove it, or the kernel and the board fight over the wire.

## Symptom

`journalctl -k` fills with SysRq dumps:

```
sysrq: HELP : loglevel(0-9) reboot(b) crash(c) terminate-all-tasks(e) ...
sysrq: This sysrq operation is disabled.
```

## Cause

`/dev/ttyAMA0` is both the kernel console and the Rosman5 link.

| Source | Setting |
|--------|---------|
| `cmdline.txt` | `console=serial0,115200`, firmware resolves to `ttyAMA0` |
| `board.py` | `/dev/ttyAMA0` at 1000000 baud |

A serial console treats a BREAK plus the next byte as a SysRq command. The 115200 vs 1M baud mismatch makes the receiver latch all-zero frames with framing errors, electrically indistinguishable from BREAK. The next mis-decoded servo byte is taken as the command. Bytes that match nothing print the HELP list.

`kernel.sysrq` is 176 on Ubuntu, which is 128 + 32 + 16:

| Bit | Enables | Live bytes |
|-----|---------|------------|
| 128 | reboot / poweroff | `b` 0x62, `o` 0x6F |
| 32 | remount read-only | `u` 0x75 |
| 16 | sync | `s` 0x73 |

Those are ordinary bytes in a servo payload. A stray hit reboots the Pi or drops the rootfs to read-only mid-run.

It breaks the other direction too: the console writes kernel log text out TX into the board's RX.

## Fix

The boot partition is A/B (`autoboot.txt` sets `tryboot_a_b=1`), so patch both slots:

```bash
sudo sed -i.bak 's/console=serial0,115200 //' \
  /boot/firmware/current/cmdline.txt \
  /boot/firmware/old/cmdline.txt
```

Both files should then read:

```
multipath=off dwc_otg.lpm_enable=0 console=tty1 root=LABEL=writable rootfstype=ext4 panic=10 rootwait fixrtc
```

Reboot to apply.

## Verify

```bash
cat /proc/consoles              # tty1 only
grep -c ttyAMA0 /proc/cmdline   # 0
ls -l /dev/ttyAMA0              # still present
journalctl -k | grep -c sysrq   # 0
```

## Notes

The API is unaffected. `/dev/ttyAMA0` comes from `enable_uart=1` in `config.txt`, not from `console=`. Dropping the console only stops the kernel from claiming the port.

You lose serial console recovery. `console=tty1` keeps boot output on HDMI.

`config.txt` sets `os_prefix=new/` under `[tryboot]`. An OS update can stage a `new/` slot carrying the stock cmdline with `console=serial0,115200` back in. Re-check after kernel updates.

Reading `dmesg` needs root here (`kernel.dmesg_restrict=1`). Use `journalctl -k` instead, which works via `adm` group membership.
