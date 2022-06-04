dydt = @(t, y) 0.3*y + t;
[tans, yans] = backward_euler(dydt, 0:0.1:1, 1);
yans(end)

function [t, y] = backward_euler(odefun,tspan,y0)
    dt = tspan(length(tspan))-tspan(length(tspan)-1); % Calculate dt from the t values
    y = zeros(length(tspan), 1); % Setup our solution column vector
    y(1) = y0; % Define the initial condition
    for k = 1:length(y)-1
        g = @(x) x - y(k) - dt*odefun(tspan(k+1), x);
        y(k+1)=fzero(g, y(k));
    end
    t = tspan;
end

function [t,y] = forward_euler(odefun,tspan,y0)
    % Forward Euler method
    % Solves the differential equation y' = f(t,y) at the times
    % specified by the vector tspan and with initial condition y0.
    %  - odefun is an anonymous function of the form odefun = @(t, v) ...
    %  - tspan is a row or column vector
    %  - y0 is a number

    dt = tspan(2)-tspan(1); % Calculate dt from the t values
    y = zeros(length(tspan), 1); % Setup our solution column vector
    y(1) = y0; % Define the initial condition
    for k = 1:length(y)-1
        y(k+1) = y(k) + dt*odefun(tspan(k), y(k)); % Forward Euler step
    end
    t = tspan;
end
