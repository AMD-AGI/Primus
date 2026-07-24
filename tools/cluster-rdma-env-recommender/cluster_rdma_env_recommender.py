import os
import glob
import subprocess
import re
from typing import Tuple
import argparse

LIB_SEARCH_PATHS = [
    "/usr/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
    "/etc/libibverbs.d",
]

class RDMAClusterMapper:
    """RDMA cluster mapping utility for environment recommendation."""

    def __init__(self):
        self.rdma_devices = []

    def get_pci_device(self, device_path: str) -> str:
        try:
            link = os.path.join(device_path, "device")
            if os.path.islink(link):
                return os.path.basename(os.readlink(link))
        except OSError:
            pass
        return "UNKNOWN_PCI"

    def find_netdev_for_pci(self, target_pci: str) -> str:
        for netdev in glob.glob("/sys/class/net/*"):
            try:
                link = os.path.join(netdev, "device")
                if os.path.islink(link):
                    if os.path.basename(os.readlink(link)) == target_pci:
                        return os.path.basename(netdev)
            except OSError:
                pass
        return "NO_NETDEV"

    def get_socket_ifname_value(self) -> str:
        try:
            cmd = "ip route show default | awk '{print $5}'"
            out = subprocess.check_output(cmd, shell=True, text=True).strip()
            if not out:
                print ("\n WARNING: no default route interfaces found.")

            ifnames = list(dict.fromkeys(out.splitlines()))
            return ifnames[0]

        except Exception as e:
            return "NA"

    # ------------------------
    # vendor mapping from pci
    # ------------------------
    def rdma_vendor_from_pci(self, pci: str) -> str:
        try:
            pci_updated = pci.replace("0000:", "")
            out = subprocess.check_output(
                ["lspci", "-s", pci, "-nn"],
                text=True
            ).lower()

            if "pensando" in out:
                return "AINIC"
            elif "broadcom" in out:
                return "BNXT"
            elif "mellanox" in out:
                return "MLNX"
            else:
                return "UNKNOWN"
        except Exception:
            pass

        return "UNKNOWN"

    # -------------------------
    # ibv devinfo parsing.
    # -------------------------
    def _ibv_devinfo(self, rdma: str) -> str:
        """Run ibv_devinfo once per device."""
        try:
            result = subprocess.run(
                ["ibv_devinfo", "-d", rdma, "-v"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return ""

    def get_firmware_version(self, output: str) -> str:
        """Extract fw_ver from ibv_devinfo output."""
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("fw_ver:"):
                return line.split("fw_ver:", 1)[1].strip()
        return "UNKNOWN"

    def get_gid_info(self, output: str) -> Tuple[str, str]:
        """Extract IPv4-mapped RoCE GID."""
        for line in output.splitlines():
            if "::ffff:" in line and "GID[" in line:
                idx = re.search(r"GID\[\s*(\d+)\]", line)
                ip = re.search(r"(::ffff:[0-9.]+)", line)
                if idx and ip:
                    return idx.group(1), ip.group(1)
        return "-", "N/A"

    # -------------------------
    # Check RDMA devices
    # -------------------------
    def scan_rdma_devices(self):
        rdma_paths = sorted(glob.glob("/sys/class/infiniband/*"))
        if not rdma_paths:
            print ("No RDMA devices found.")
            return

        for path in rdma_paths:
            rdma = os.path.basename(path)
            pci = self.get_pci_device(path)
            netdev = self.find_netdev_for_pci(pci)

            ibv_out = self._ibv_devinfo(rdma)
            firmware = self.get_firmware_version(ibv_out)
            gid_index, gid_value = self.get_gid_info(ibv_out)
            vendor = self.rdma_vendor_from_pci(pci)

            self.rdma_devices.append({
                "rdma": rdma,
                "pci": pci,
                "netdev": netdev,
                "firmware": firmware,
                "gid_index": gid_index,
                "gid_value": gid_value,
                "vendor": vendor,
            })

    # -------------------------
    # REPORTING
    # -------------------------
    def print_table(self):
        print ("\n RDMA CLUSTER MAPPING")
        print ("=" * 100)
        print (f"{'RDMA':<8} | {'PCI':<12} | {'NETDEV':<10} | {'FIRMWARE':<15} | {'GID_IDX':<7} | {'GID':<20} | VENDOR ")
        print ("-" * 100)

        for d in self.rdma_devices:
            print (
                f"{d['rdma']:<8} | {d['pci']:<12} | {d['netdev']:<10} | "
                f"{d['firmware']:<15} | {d['gid_index']:<7} | {d['gid_value']:<20} | {d['vendor']}"
            )

    def print_detailed_info(self):
        print ("\n DETAILED RDMA DEVICE ANALYSIS")
        print ("=" * 80)

        for d in self.rdma_devices:
            print (f"\nDevice: {d['rdma']}")
            print (f"  PCI:       {d['pci']}")
            print (f"  NETDEV:    {d['netdev']}")
            print (f"  Firmware:  {d['firmware']}")
            print (f"  GID Index: {d['gid_index']}")
            print (f"  GID:       {d['gid_value']}")
            #print (f"  Status:    {self._analyze_firmware_version(d['firmware'])}")
            print (f"  VENDOR:    {d['vendor']}")

    def _analyze_firmware_version(self, fw: str) -> str:
        """
            TODO: analyze the firmware version and recommend the best based on NIC.
        """

    def generate_firmware_report(self):
        print ("\n FIRMWARE VERSION REPORT")
        print ("=" * 60)

        fw_map = {}
        for d in self.rdma_devices:
            fw_map.setdefault(d["firmware"], []).append(d["rdma"])

        for fw, devs in fw_map.items():
            print (f"\nFirmware: {fw}")
            #print (f"Status:   {self._analyze_firmware_version(fw)}")
            print (f"Devices:  {', '.join(devs)}")

        if len(fw_map) > 1:
            print ("\n Multiple firmware versions detected — standardization recommended")

    def _find_lib(self, patterns):
        """ Find first matching library for given glob patterns."""
        for base in LIB_SEARCH_PATHS:
            for pat in patterns:
                matches = glob.glob(os.path.join(base, "**", pat), recursive=True)
                if matches:
                    return matches[0]

        return None

    def _find_all_libs(self, patterns):
        """Find all matching libraries."""
        found = []
        for base in LIB_SEARCH_PATHS:
            for pat in patterns:
                found.extend(glob.glob(os.path.join(base, "**", pat), recursive=True))
        return sorted(set(found))

    def _docker_cmd_bnxt(self):
        bnxt_rdma = self._find_lib(["libbnxt_re-rdmav*.so"])
        rdmacm = self._find_lib(["librdmacm.so.1"])
        ibverbs = self._find_lib(["libibverbs.so.1"])
        libnl3 = self._find_lib(["libnl-3.so.200"])
        libnl3_router = self._find_lib(["libnl-route-3.so.200"])

        if not bnxt_rdma:
            print ("Missing libbnxt_re-rdma*.so files \n")
        if not rdmacm:
            print ("Missing librdmacm.so* files \n")
        if not libnl3:
            print ("Missing libnl* files \n")

        print ("Libraries detected on host device:")
        print (f"{bnxt_rdma:>5}")
        print (f"{rdmacm:>5}")
        print (f"{ibverbs:>5}")
        print (f"{libnl3:>5}")
        print (f"{libnl3_router:>5}")

        cmd_string = f"""
            docker run --rm -it \\
                --device /dev/dri \\
                --device /dev/infiniband \\
                --device /dev/kfd \\
                --network host \\
                --ipc host \\
                --privileged \\
                --ulimit memlock=-1:-1 \\
                --group-add video \\
                --cap-add SYS_PTRACE \\
                --security-opt seccomp=unconfined \\
                --shm-size 64G \\
                -v /sys:/sys \\
                -v $HOME/.ssh:/root/.ssh \\
                -v $HOME:$HOME \\
                -v /dev/infiniband:/dev/infiniband \\
                -v /sys/class/infiniband:/sys/class/infiniband:ro \\
                -v /sys/class/net:/sys/class/net:ro \\
                -v /sys/bus/pci:/sys/bus/pci:ro \\
                -v /etc/libibverbs.d:/etc/libibverbs.d:ro \\
                -v /etc/rdma:/etc/rdma:ro \\
        """
        if bnxt_rdma:
            cmd_string = cmd_string + (" " * 8) + f"-v {bnxt_rdma}:{bnxt_rdma}:ro \\\n"
        if rdmacm:
            cmd_string = cmd_string + (" " * 16) + f"-v {rdmacm}:{rdmacm}:ro \\\n"
        if ibverbs:
            cmd_string = cmd_string + (" " * 16) + f"-v {ibverbs}:{ibverbs}:ro \\\n"
        if libnl3:
            cmd_string = cmd_string + (" " * 16) + f"-v {libnl3}:{libnl3}:ro \\\n"
        if libnl3_router:
            cmd_string = cmd_string + (" " * 16) + f"-v {libnl3_router}:{libnl3_router}:ro \\\n"

        cmd_string = cmd_string + (" " * 16) + "<image> \n"

        return cmd_string.strip()

    def _docker_cmd_mlnx(self):
        cmd_string = f"""
            docker run --rm -it \\
                --device /dev/dri \\
                --device /dev/infiniband \\
                --device /dev/kfd \\
                --network host \\
                --ipc host \\
                --privileged \\
                --ulimit memlock=-1:-1 \\
                --group-add video \\
                --cap-add SYS_PTRACE \\
                --security-opt seccomp=unconfined \\
                --shm-size 64G \\
                -v /sys:/sys \\
                -v $HOME/.ssh:/root/.ssh \\
                -v $HOME:$HOME \\
                -v /dev/infiniband:/dev/infiniband \\
                -v /sys/class/infiniband:/sys/class/infiniband:ro \\
                -v /sys/class/net:/sys/class/net:ro \\
                -v /sys/bus/pci:/sys/bus/pci:ro \\
        """
        cmd_string = cmd_string + (" " * 8) + "<image> \n"
        return cmd_string.strip()

    def _docker_cmd_ionic(self):
        ionic_rdma = self._find_lib(["libionic-rdmav*.so"])
        ionic_so = self._find_all_libs(["libionic.so*"])
        ionic_driver = self._find_lib(["ionic.driver"])

        if not ionic_rdma:
            print ("Missing libionic-rdma*.so files \n")
        if not ionic_so:
            print ("Missing libionic.so files \n")
        if not ionic_driver:
            print ("Missing ionic.driver file \n")

        print ("\n")
        print ("Libraries detected on host device:")
        print (f"{ionic_rdma:>5}")
        print (f"{ionic_driver:>5}")
        for so_file in ionic_so:
            print (f"{so_file:>5}")

        cmd_string = f"""
            docker run --rm -it \\
                --device /dev/dri \\
                --device /dev/infiniband \\
                --device /dev/kfd \\
                --network host \\
                --ipc host \\
                --privileged \\
                --ulimit memlock=-1:-1 \\
                --group-add video \\
                --cap-add SYS_PTRACE \\
                --security-opt seccomp=unconfined \\
                --shm-size 64G \\
                -v /sys:/sys \\
                -v $HOME/.ssh:/root/.ssh \\
                -v $HOME:$HOME \\
                -v /dev/infiniband:/dev/infiniband \\
                -v /sys/class/infiniband:/sys/class/infiniband:ro \\
                -v /sys/class/net:/sys/class/net:ro \\
                -v /sys/bus/pci:/sys/bus/pci:ro \\
        """

        if ionic_rdma:
            cmd_string = cmd_string + (" " * 8) + f"-v {ionic_rdma}:{ionic_rdma}:ro \\\n"
        if ionic_so:
            for j in range(len(ionic_so)):
                so_file = ionic_so[j]
                cmd_string = cmd_string + (" " * 16) + f"-v {so_file}:{so_file}:ro \\\n"
        if ionic_driver:
            cmd_string = cmd_string + (" " * 16) + f"-v {ionic_driver}:{ionic_driver}:ro \\\n"

        cmd_string = cmd_string + (" " * 16) + "<image> \n"

        return cmd_string.strip()

    def generate_docker_launch_command(self):
        vendors = {d['vendor'] for d in self.rdma_devices}

        print ("\n RECOMMENDED DOCKER LAUNCH COMMAND")
        print ("=" * 80)

        if len(vendors) > 1:
            print ("\n WARNING: Multiple RDMA vendors detected.")

        print ("\n")
        print ("Vendors detected: {}".format(vendors))
        docker_cmd = ""
        if "AINIC" in vendors:
            docker_cmd = self._docker_cmd_ionic()
            print ("\n")
            print ("Docker launch command:")
            print (docker_cmd)
        elif "BNXT" in vendors:
            docker_cmd = self._docker_cmd_bnxt()
            print ("\n")
            print ("Docker launch command:")
            print (docker_cmd)
        elif "MLNX" in vendors:
            docker_cmd = self._docker_cmd_mlnx()
            print ("\n")
            print ("Docker launch command:")
            print (docker_cmd)
        else:
            print ("\n WARNING: this vendor docker cannot be generated, please verify manually.")

        return docker_cmd

    def _get_nccl_env_variables(self):
        nccl_env_variables = []

        # IB NCCL CPU affinity.
        nccl_env_variables.append("export NCCL_IGNORE_CPU_AFFINITY=1")
        # IB GID
        gid_indexes = {d['gid_index'] for d in self.rdma_devices}
        if (len(gid_indexes) > 1):
            print (" \n WARNING: multiple GID indeces detected, please check detailed report for mapping the env variables.")
        nccl_env_variables.append(f"export NCCL_IB_GID_INDEX={max(list(gid_indexes))}")

        # IB HCA
        #rdma_devices = ",".join([d['rdma'] for d in self.rdma_devices])
        rdma_devices=""
        firmware_version = max([d['firmware'] for d in self.rdma_devices])
        for d in self.rdma_devices:
            device = d['rdma'] if d['gid_index'].isnumeric() and d['firmware'] == firmware_version else ""
            if device:
                rdma_devices = rdma_devices + device + ","
        nccl_env_variables.append(f"export NCCL_IB_HCA={rdma_devices.rstrip(',')}")

        # NCCL/GLOO socket if name if framework requires.
        socket_ifname = self.get_socket_ifname_value()
        nccl_env_variables.append(f"export NCCL_SOCKET_IFNAME={socket_ifname}")
        nccl_env_variables.append(f"export GLOO_SOCKET_IFNAME={socket_ifname}")

        return nccl_env_variables

    def generate_framework_env_variables(self):
        print ("\n RECOMMENDED ENV VARIABLES")
        print ("=" * 80)

        env_variables = []
        print ("Note: Please cross check before exporting in your scripts")
        nccl_env_variables = self._get_nccl_env_variables()
        print ("\n NCCL Env Variables:")
        for var in nccl_env_variables:
            print (var)

        print ("\n rocSHMEM Env variables:")
        rocshmem_env =[]
        rocshmem_env.append("export ROCSHMEM_HEAP_SIZE=7524589824")
        rocshmem_env.append("export ROCSHMEM_MAX_NUM_CONTEXTS=256")
        for var in rocshmem_env:
            print (var)

        env_variables = nccl_env_variables + rocshmem_env
        return env_variables

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", help="Generate HTML report", action="store_true")
    args = parser.parse_args()

    mapper = RDMAClusterMapper()
    mapper.scan_rdma_devices()

    if not mapper.rdma_devices:
        return

    mapper.print_table()
    mapper.print_detailed_info()
    mapper.generate_firmware_report()
    mapper.generate_docker_launch_command()
    mapper.generate_framework_env_variables()

if __name__ == "__main__":
    main()
