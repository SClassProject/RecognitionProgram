import socketio
from id import room_id, id

sio = socketio.Client(ssl_verify=False)

@sio.on('connect', namespace='/room')
def connect() :
    print('Connection established')
    sio.emit('join', {'id': id, 'room_id' : room_id}, namespace='/room')

@sio.on('attend', namespace='/room')
def attend(data) :
    sio.emit('attend', {'msg': data}, namespace='/room')

@sio.on('hand', namespace='/room')
def raiseHand(data) :
    sio.emit('hand', {'msg': data}, namespace='/room')

@sio.on('disconnect', namespace='/room')
def disconnect() :
    print('Disconnected from server')
    
    