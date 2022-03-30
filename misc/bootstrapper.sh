#!/bin/bash
set -x -e

echo -e 'export PYTHONPATH=/usr/bin/python3
export PYSPARK_PYTHON=/usr/bin/python3
export HADOOP_CONF_DIR=/etc/hadoop/conf
export SPARK_JARS_DIR=/usr/lib/spark/jars
export SPARK_HOME=/usr/lib/spark,/usr/lib64/spark' >> $HOME/.bashrc && source $HOME/.bashrc

sudo python3 -m pip install awscli boto spark-nlp matplotlib numpy==1.17

set +x
exit 0