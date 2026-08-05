# Cluster RDMA Environment Mapping & Recommender Tool

Cluster RDMA Env Mapping & Recommender tool is an end to end RDMA environment discovery and recommendation tool that automatically inspects the host, maps the RDMA devices to physical NICs and generates production ready recommendations. This tool bridges the critical gap between bare metal RDMA hardware and Distributed Inference workloads by reducing manual trial and error. 

The tool contains the following features:

* RDMA → PCI → NetDev mapping

* Vendor Detection (AINIC/BNXT/MLNX)

* Firmware & RoCEv2 GID Discovery

* Recommended docker launch command.

* Recommended Framework level environment variables required for workloads.

* Human readable CLI report

## Steps to run the tool

On baremetal, run the following (Currently only works for Ubuntu)

```
cd tools/cluster-rdma-env-recommender
python3 cluster_rdma_env_recommender.py |& tee report.txt
```

The report.txt contains the details of the above mentioned features of a cluster.




