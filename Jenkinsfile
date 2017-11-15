#!/usr/bin/groovy
// TOOD: rename to @Library('h2o-jenkins-pipeline-lib') _
@Library('test-shared-library') _

import ai.h2o.ci.Utils

def utilsLib = new Utils()

def SAFE_CHANGE_ID = changeId()
def CONTAINER_NAME

String changeId() {
    if (env.CHANGE_ID) {
        return "-${env.CHANGE_ID}".toString()
    }
    return "-master"
}

pipeline {
    agent none

    // Setup job options
    options {
        ansiColor('xterm')
        timestamps()
        timeout(time: 120, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
        skipDefaultCheckout()
    }

    environment {
        MAKE_OPTS = "-s CI=1" // -s: silent mode
    }

    stages {

        stage('Build on Linux CUDA8 NCCL') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }

                script {
                    CONTAINER_NAME = "xgboost${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/xgboost-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/xgboost-build
                                    nvidia-docker exec ${
                                CONTAINER_NAME
                            } bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; make ${
                                env.MAKE_OPTS
                            } AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -f Makefile2 libxgboost ; rm -rf build/VERSION.txt ; make -f Makefile2 build/VERSION.txt'
                                """
                            stash includes: 'python-package/dist/*.whl', name: 'linux_whl'
                            stash includes: 'build/VERSION.txt', name: 'version_info'
                            // Archive artifacts
                            arch 'python-package/dist/*.whl'
                        } finally {
                            sh "nvidia-docker stop ${CONTAINER_NAME}"
                        }
                    }
                }
            }
        }
        stage('Test on Linux CUDA8 NCCL') {
            agent {
                label "gpu && nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }
            steps {
                dumpInfo 'Linux Test Info'
                // Get source code (should put tests into wheel, then wouldn't have to checkout)
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    checkout scm
                }
                unstash 'linux_whl'
                script {
                    try {
                        sh """
                            nvidia-docker run  --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/xgboost-build
                            nvidia-docker exec ${CONTAINER_NAME} bash -c 'export HOME=`pwd`; eval \"\$(/root/.pyenv/bin/pyenv init -)\"  ; /root/.pyenv/bin/pyenv global 3.6.1; pip install `find python-package/dist -name "*xgboost*.whl"`; mkdir -p build/test-reports/ ; python -m nose --with-xunit --xunit-file=build/test-reports/xgboost.xml tests/python-gpu'
                        """
                    } finally {
                        sh """
                            nvidia-docker stop ${CONTAINER_NAME}
                        """
                        junit testResults: 'build/test-reports/*.xml', keepLongStdio: true, allowEmptyResults: false
                        deleteDir()
                    }
                }
            }
        }

        stage('Publish to S3 CUDA8 NCCL') {
            agent {
                label "linux"
            }

            steps {
                unstash 'linux_whl'
                unstash 'version_info'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    sh 'echo "Stashed files:" && ls -l python-package/dist/'
               script {
                    // Load the version file content
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    version = null // This is necessary, else version:Tuple will be serialized

                    if (isRelease()) {
                        def artifact = "xgboost-${versionTag}-py3-none-any.whl"
                        def localArtifact = "python-package/dist/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/stable/ai/h2o/xgboost/${versionTag}/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }

                    if (isBleedingEdge()) {
                        def artifact = "xgboost-${versionTag}-py3-none-any.whl"
                        def localArtifact = "python-package/dist/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/${versionTag}/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }
                }
 
                    }
                }
            }
        }


        stage('Build on Linux CUDA8 noNCCL') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }

                script {
                    CONTAINER_NAME = "xgboost${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/xgboost-build -f Dockerfile-build --build-arg cuda=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/xgboost-build
                                    nvidia-docker exec ${
                                CONTAINER_NAME
                            } bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; make ${
                                env.MAKE_OPTS
                            } AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -f Makefile2 libxgboost2 ; rm -rf build/VERSION.txt ; make -f Makefile2 build/VERSION.txt ; mkdir -p python-package/dist2 ; mv python-package/dist/*.whl python-package/dist2/'
                                """
                            stash includes: 'python-package/dist2/*.whl', name: 'linux_whl'
                            stash includes: 'build/VERSION.txt', name: 'version_info'
                            // Archive artifacts
                            arch 'python-package/dist2/*.whl'
                        } finally {
                            sh "nvidia-docker stop ${CONTAINER_NAME}"
                        }
                    }
                }
            }
        }

        stage('Publish to S3 CUDA8 noNCCL') {
            agent {
                label "linux"
            }

            steps {
                unstash 'linux_whl'
                unstash 'version_info'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    sh 'echo "Stashed files:" && ls -l python-package/dist2/'
               script {
                    // Load the version file content
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    version = null // This is necessary, else version:Tuple will be serialized

                    if (isRelease()) {
                        def artifact = "xgboost-${versionTag}-py3-none-any.whl"
                        def localArtifact = "python-package/dist2/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/stable/ai/h2o/xgboost/${versionTag}_nonccl_cuda8/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }

                    if (isBleedingEdge()) {
                        def artifact = "xgboost-${versionTag}-py3-none-any.whl"
                        def localArtifact = "python-package/dist2/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/${versionTag}_nonccl_cuda8/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }
                }
                    }
                }
            }
        }

        stage('Build on Linux CUDA9 noNCCL') {
            agent {
                label "nvidia-docker && (mr-dl11||mr-dl16||mr-dl10)"
            }

            steps {
                dumpInfo 'Linux Build Info'
                // Do checkout
                retryWithTimeout(100 /* seconds */, 3 /* retries */) {
                    deleteDir()
                    checkout([
                            $class                           : 'GitSCM',
                            branches                         : scm.branches,
                            doGenerateSubmoduleConfigurations: false,
                            extensions                       : scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: true, recursiveSubmodules: false, reference: '', trackingSubmodules: false, shallow: true]],
                            submoduleCfg                     : [],
                            userRemoteConfigs                : scm.userRemoteConfigs])
                }

                script {
                    CONTAINER_NAME = "xgboost${SAFE_CHANGE_ID}-${env.BUILD_ID}"
                    // Get source code
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                        try {
                            sh """
                                    nvidia-docker build  -t opsh2oai/xgboost-build -f Dockerfile-build --build-arg cuda=cuda:9.0-cudnn7-devel-ubuntu16.04 .
                                    nvidia-docker run --init --rm --name ${CONTAINER_NAME} -d -t -u `id -u`:`id -g` -w `pwd` -v `pwd`:`pwd`:rw --entrypoint=bash opsh2oai/xgboost-build
                                    nvidia-docker exec ${
                                CONTAINER_NAME
                            } bash -c 'eval \"\$(/root/.pyenv/bin/pyenv init -)\" ; /root/.pyenv/bin/pyenv global 3.6.1; make ${
                                env.MAKE_OPTS
                            } AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -f Makefile2 libxgboost2 ; rm -rf build/VERSION.txt ; make -f Makefile2 build/VERSION.txt ; mkdir -p python-package/dist3 ; mv python-package/dist/*.whl python-package/dist3/'
                                """
                            stash includes: 'python-package/dist3/*.whl', name: 'linux_whl'
                            stash includes: 'build/VERSION.txt', name: 'version_info'
                            // Archive artifacts
                            arch 'python-package/dist3/*.whl'
                        } finally {
                            sh "nvidia-docker stop ${CONTAINER_NAME}"
                        }
                    }
                }
            }
        }

        stage('Publish to S3 CUDA9 noNCCL') {
            agent {
                label "linux"
            }

            steps {
                unstash 'linux_whl'
                unstash 'version_info'
                retryWithTimeout(200 /* seconds */, 5 /* retries */) {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: "awsArtifactsUploader"]]) {
                    sh 'echo "Stashed files:" && ls -l python-package/dist3/'
               script {
                    // Load the version file content
                    def versionTag = utilsLib.getCommandOutput("cat build/VERSION.txt | tr '+' '-'")
                    version = null // This is necessary, else version:Tuple will be serialized

                    if (isRelease()) {
                        def artifact = "xgboost-${versionTag}-py3-none-any.whl"
                        def localArtifact = "python-package/dist3/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/stable/ai/h2o/xgboost/${versionTag}_nonccl_cuda8/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }

                    if (isBleedingEdge()) {
                        def artifact = "xgboost-${versionTag}-py3-none-any.whl"
                        def localArtifact = "python-package/dist3/${artifact}"
                        def bucket = "s3://artifacts.h2o.ai/releases/bleeding-edge/ai/h2o/xgboost/${versionTag}_nonccl_cuda8/"
                        sh "s3cmd put ${localArtifact} ${bucket}"
                        sh "s3cmd setacl --acl-public  ${bucket}${artifact}"
                    }
                }
                    }
                }
            }
        }

    }
    post {
        failure {
            node('mr-dl11') {
                script {
                    // Hack - the email plugin finds 0 recipients for the first commit of each new PR build...
                    def email = utilsLib.getCommandOutput("git --no-pager show -s --format='%ae'")
                    emailext(
                            to: "jmckinney@h2o.ai",
                            subject: "BUILD FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                            body: '''${JELLY_SCRIPT, template="html_gmail"}''',
                            attachLog: true,
                            compressLog: true,
                    )
                }
            }
        }
    }
}

def isRelease() {
    return env.BRANCH_NAME.startsWith("rel")
}

def isBleedingEdge() {
    return env.BRANCH_NAME.startsWith("h2oai")
}
