 
 
 
 Scalable-Data-Pipeline oluşturmak için gerekli docker image'leri oluşturur ve çalıştırır.
 
 Docker Images: 
                 mongodb    
                iot-agent   
                orion      
                crate-db    
                quantumleap 
                tutorial    
                grafana     
 
 Image'lerin Oluşturulma ve Çalıştırılması

    ---- ./services create komutu ile pull işlemi gerçekleşir.
    ---- ./services start komutu ile oluşturulan image'ler başlatılır.

    Bu işlemler gerçekleştirildiğinde import-data dosyası çift tıklanır veya 'bash import-data' komutu ile çalıştırılır.
    
 import-data dosyası: Bu dosya içerisinde ;
    
    - Eskişehir lokasyonunda tanımlı lab1
        -  curl -iX POST \
            'http://localhost:1026/v2/entities' \
            -H 'Content-Type: application/json' \
            -d '
            {
                "id": "urn:ngsi-ld:Lab:001",
                "type": "Lab",
                "address": {
                    "type": "PostalAddress",
                    "value": {
                        "streetAddress": "Gençlik Bulvarı",
                        "addressRegion": "Eskişehir",
                        "addressLocality": "Eskişehir Osmangazi Üniversitesi",
                        "postalCode": "26040"
                    },
                    "metadata": {
                        "verified": {
                            "value": true,
                            "type": "Boolean"
                        }
                    }
                },
                "location": {
                    "type": "geo:json",
                    "value": {
                        "type": "Point",
                        "coordinates": [39.74928562191364,30.48216550271973]
                    }
                },
                "name": {
                    "type": "Text",
                    "value": "ASIST"
                }
            }'

    - Lab1 içinde tanımlı motor1 ve motor2
        - curl -iX POST \
            'http://localhost:1026/v2/op/update' \
            -H 'Content-Type: application/json' \
            -d '{
            "actionType":"APPEND",
            "entities":[
                {
                "id":"urn:ngsi-ld:Motor:unit001", "type":"Motor",
                "location":{
                    "type":"geo:json", "value":{ "type":"Point","coordinates":[39.74928562191364,30.48216550271973]}
                },
                "name":{
                    "type":"Text", "value":"Motor1"
                },
                "vibrationSensorCount":{
                    "type":"Integer", "value":1
                }
                },
                {
                "id":"urn:ngsi-ld:Motor:unit002", "type":"Motor",
                "location":{
                    "type":"geo:json","value":{"type":"Point","coordinates":[39.74928562191364,30.48216550271973]}
                },
                "name":{
                    "type":"Text", "value":"Motor2"
                },
                "vibrationSensorCount":{
                    "type":"Integer", "value":1
                }
                }
            ]
            }'

    - Motor1 içinde tanımlı Vibrasyon Sensör1, sensörün nitelikleri aşağıdaki şekilde tanımlanmıştır.
        - curl -iX POST \
            'http://localhost:4041/iot/devices' \
            -H 'Content-Type: application/json' \
            -H 'fiware-service: openiot' \
            -H 'fiware-servicepath: /' \
            -d '{
            "devices": [
            {
                "device_id":   "vibrationSensor001",
                "entity_name": "urn:ngsi-ld:VibrationSensor:001",
                "entity_type": "VibrationSensor",
                "timezone":    "Europe/Berlin",
                "attributes": [
                { "object_id": "MIN_x", "name": "MIN_x", "type": "Double" },
                { "object_id": "MIN_y", "name": "MIN_y", "type": "Double" },
                { "object_id": "MIN_z", "name": "MIN_z", "type": "Double" },
                { "object_id": "MAX_x", "name": "MAX_x", "type": "Double" },
                { "object_id": "MAX_y", "name": "MAX_y", "type": "Double" },
                { "object_id": "MAX_z", "name": "MAX_z", "type": "Double" },
                { "object_id": "MEAN_x", "name": "MEAN_x", "type": "Double" },
                { "object_id": "MEAN_y", "name": "MEAN_y", "type": "Double" },
                { "object_id": "MEAN_z", "name": "MEAN_z", "type": "Double" },
                { "object_id": "RMS_x", "name": "RMS_x", "type": "Double" },
                { "object_id": "RMS_y", "name": "RMS_y", "type": "Double" },
                { "object_id": "RMS_z", "name": "RMS_z", "type": "Double" },
                { "object_id": "STD_x", "name": "STD_x", "type": "Double" },
                { "object_id": "STD_y", "name": "STD_y", "type": "Double" },
                { "object_id": "STD_z", "name": "STD_z", "type": "Double" },
                { "object_id": "SKEW_x", "name": "SKEW_x", "type": "Double" },
                { "object_id": "SKEW_y", "name": "SKEW_y", "type": "Double" },
                { "object_id": "SKEW_z", "name": "SKEW_z", "type": "Double" },
                { "object_id": "KURT_x", "name": "KURT_x", "type": "Double" },
                { "object_id": "KURT_y", "name": "KURT_y", "type": "Double" },
                { "object_id": "KURT_z", "name": "KURT_z", "type": "Double" }
                ],
                "static_attributes": [
                { "name":"refMotor", "type": "Relationship", "value": "urn:ngsi-ld:Motor:unit001"}
                ]
                }
                ]
            }'
            
    - Vibrasyon sensörü için veri atıldığında CrateDb'ye yazılması için QuantumLeap ile configurasyon
        - curl -iX POST \
            'http://localhost:1026/v2/subscriptions/' \
            -H 'Content-Type: application/json' \
            -H 'fiware-service: openiot' \
            -H 'fiware-servicepath: /' \
            -d '{
            "description": "Notify QuantumLeap of count changes of any Vibration Sensor",
            "subject": {
                "entities": [
                {
                    "idPattern": "VibrationSensor.*"
                }
                ],
                "condition": {
                "attrs": [
                "maxx",
                "MIN_x",
                "MIN_y",
                "MIN_z",
                "MAX_x",
                "MAX_y",
                "MAX_z",
                "MEAN_x",
                "MEAN_y",
                "MEAN_z",
                "RMS_x", 
                "RMS_y",
                "RMS_z",
                "STD_x",
                "STD_y",
                "STD_z",
                "SKEW_x",
                "SKEW_y",
                "SKEW_z",
                "KURT_x",
                "KURT_y",
                "KURT_z"
                ]
                }
            },
            "notification": {
                "http": {
                "url": "http://quantumleap:8668/v2/notify"
                },
                "attrs": [
                "MIN_x",
                    "MIN_y",
                    "MIN_z",
                    "MAX_x",
                    "MAX_y",
                    "MAX_z",
                    "MEAN_x",
                    "MEAN_y",
                    "MEAN_z",
                    "RMS_x", 
                    "RMS_y",
                    "RMS_z",
                    "STD_x",
                    "STD_y",
                    "STD_z",
                    "SKEW_x",
                    "SKEW_y",
                    "SKEW_z",
                    "KURT_x",
                    "KURT_y",
                    "KURT_z"
                ],
                "metadata": ["dateCreated", "dateModified"]
            },
            "throttling": 1
            }'
        
    - Configurasyon tanımını kontrol edebilmek için gönderilen 2 adet veri gönderme komutu yer almaktadır.
        -curl -iX POST \
            'http://localhost:7896/iot/json?k=4jggokgpepnvsb2uv4s40d59ov&i=vibrationSensor001' \
            -H 'Content-Type: application/json' \
            -d '[{"MIN_x": "0.8628"},{"MIN_y": "0.87838"},{"MIN_z": "0.23737"},{"MAX_x": "0.172893898"},{"MAX_y": "0.36486473"},{"MAX_z": "0.3685863"},{"MEAN_x": "0.1534578"},{"MEAN_y": "0.345346"},{"MEAN_z": "0.34747"},{"RMS_x": "0.36236"},{"RMS_y": "0.45734"},{"RMS_z": "0.8628"},{"STD_x": "0.87838"},{"STD_y": "0.23737"},{"STD_z": "0.172893898"},{"SKEW_x": "0.36486473"},{"SKEW_y": "0.3685863"},{"SKEW_z": "0.1534578"},{"KURT_x": "0.345346"},{"KURT_y": "0.34747"},{"KURT_z": "0.36236"}]'
    
    
    
    
    
