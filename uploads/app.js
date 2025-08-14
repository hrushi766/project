var evtSource = new EventSource("/alert_stream");

evtSource.onmessage = function(e) {
    var data = JSON.parse(e.data);

    document.getElementById('alertText').innerText = data.event;
    document.getElementById('alertLocation').innerText = data.location;
    document.getElementById('alertTime').innerText = data.timestamp;

    if(data.image){
        document.getElementById('alertImage').src = 'data:image/jpeg;base64,' + data.image;
    } else {
        document.getElementById('alertImage').src = '';
    }
};

