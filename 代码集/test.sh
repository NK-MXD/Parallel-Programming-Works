# test.sh
# !/bin/sh
# PBS -N test
# PBS -l nodes=4:ppn=1
pssh -h $PBS_NODEFILE mkdir -p /home/s2010349/myTest
pscp.pssh -h $PBS_NODEFILE /home/s2010349/myTest/MPIonARM /home/s2010349/myTest/
pssh -h $PBS_NODEFILE /tmp/noPasswordLogin.sh s2010349 s2010349
/usr/local/bin/mpiexec -np 4 -machinefile $PBS_NODEFILE /home/s2010349/myTest/MPIonARM