% Define variables and constants
base_radius = 5; % cm
offset = 0.5; % cm
amplitude = 4; % cm
dwell1_start = 0; % degrees
dwell1_end = 80; % degrees
rise_start = 80; % degrees
rise_end = 180; % degrees
dwell2_start = 180; % degrees
dwell2_end = 240; % degrees
return_start = 240; % degrees
return_end = 360; % degrees

% Define functions for each segment of the cam motion
dwell1 = @(angle) 0;
cycloidal_rise = @(angle) amplitude * ( (angle - rise_start) / (rise_end - rise_start) - sin(2 * pi * (angle - rise_start) / (rise_end - rise_start)) / (2 * pi) );
dwell2 = @(angle) 0;
simple_harmonic_return = @(angle) amplitude * sin(pi * (angle - return_start) / (return_end - return_start));

% Create an empty array to store the cam profile radii
phi = zeros(1, 361);

% Iterate over the range of cam rotation angles
for angle = 0:360
    % Determine which segment of the cam motion the current angle falls into
    if angle >= dwell1_start && angle < dwell1_end
        displacement = dwell1(angle);
    elseif angle >= rise_start && angle < rise_end
        displacement = cycloidal_rise(angle);
    elseif angle >= dwell2_start && angle < dwell2_end
        displacement = dwell2(angle);
    elseif angle >= return_start && angle <= return_end
        displacement = simple_harmonic_return(angle);
    else
        error("Angle %d is outside the range of 0 to 360 degrees.", angle);
    end
    
    % Calculate the radius of the cam at the current angle
    radius = base_radius + offset + displacement;
    
    % Add the radius to the array
    phi(angle + 1) = radius;
end

% Plot the cam profile in polar coordinates
theta = linspace(0, 2 * pi, 361);
polarplot(theta, phi);
