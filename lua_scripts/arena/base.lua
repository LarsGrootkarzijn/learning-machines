sim = require("sim")

local sensor
local food
local distance

function sysCall_init()
    sensor = sim.getObject("./Base_Proximity_sensor")
    base = sim.getObject("/Base")
    base_pos = sim.getObjectPosition(base, -1)
    food = sim.getObject("/Food")
    food_pos = sim.getObjectPosition(food, -1)
    
    base_pos[1] = ((math.random() * 2) - 4.1)
    base_pos[2] = ((math.random() * 2) - 0.2)
    
    if
        base_pos[2] > -0.2
        and base_pos[2] < 0.3 
    then
        base_pos[2] = base_pos[2] + 0.5
    end
    sim.setObjectPosition(base, -1, base_pos)
    
    food_pos[1] = ((math.random() * 2) - 4.1)
    food_pos[2] = ((math.random() * 2) - 0.2)
    
    if
        food_pos[2] > -0.2
        and food_pos[2] < 0.3 
    then
        food_pos[2] = food_pos[2] + 0.5
    end
    sim.setObjectPosition(food, -1, food_pos)
    
    distance = -1.0
end

function sysCall_actuation()
    detected, dist, points, obj, n = sim.checkProximitySensor(sensor, food)
    if detected then
        distance = dist
    else
        distance = -1.0
    end
end

getFoodDistance = function(inIntegers, inFloats, inStrings, inBuffer)
    return {}, { distance }, {}, ""
end
