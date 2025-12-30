pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/openhubber/MLOps.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t fraud-detector .'
            }
        }
        stage('Push to Private Registry') {
            steps {
                sh '''
                    REG=host.docker.internal:6000
                    docker tag fraud-detector $REG/fraud-detector:latest
                    docker push $REG/fraud-detector:latest
                   '''
            }
        }
        stage('Deploy Model') {
            steps {
                sh '''
                    REG=host.docker.internal:6000
                    docker pull $REG/fraud-detector:latest
                    docker rm -f fraud-detector || true
                    docker run -d --name fraud-detector --rm -p 8000:8080 $REG/fraud-detector:latest
                   '''
            }
        }
    }
}
