% Create the outer cube
% a 8x3 matrix
outer = [ 0.0  0.0  0.0;    % 1
          0.0  0.0  2.0;    % 2
          0.0  2.0  0.0;    % 3
          0.0  2.0  2.0;    % 4
          2.0  0.0  0.0;    % 5
          2.0  0.0  2.0;    % 6
          2.0  2.0  0.0;    % 7
          2.0  2.0  2.0 ]'; % 8

% conver to homogenous
outer_tilde = [ outer; ones(1,size(outer,2)) ];
% let the origin to be the center
trans =   [ 1.0 0.0 0.0 -1.0;
            0.0 1.0 0.0 -1.0;
            0.0 0.0 1.0 -1.0;
            0.0 0.0 0.0  1.0 ];
outer_tilde = rotMatYZ(20) * outer_tilde;
outer_tilde = trans * outer_tilde;

% create an inner cube 
generateInner =   [ 0.5 0.0 0.0 0.0;
                    0.0 0.5 0.0 0.0;
                    0.0 0.0 0.5 0.0;
                    0.0 0.0 0.0 1.0 ];
inner_tilde = generateInner * outer_tilde;

% Create a new figure dialog and set the background color to white.
figure;
set(gcf, 'color','w');
set(gcf, 'Position', [200, 200, 800, 600])
axis([-2 2 -2 2])

% Show original shape in blue
showCube(outer_tilde,'k');
showCube(inner_tilde,'k');
connect(outer_tilde, inner_tilde, 'k');
view(43,12)

%outer_tilde = rot_mat * outer_tilde;
%showCube(outer_tilde,'b');
%showCube(inner_tilde,'r');
formattype = 'png';
saveas(gcf,num2str(0),formattype)
view(40,12)

clf(gcf)
for k = 1:3:360
    % rotation matrix
    rot_matXY = rotMatXY(k);
    % scaling
    val1 = sind(k) + 1;
    scaling1 = [ val1 0.0  0.0   0.0;
                 0.0  val1 0.0   0.0;
                 0.0  0.0  val1  0.0;
                 0.0  0.0  0.0   1.0 ];
    val2 = -sind(k) + 1;
    scaling2 = [ val2 0.0  0.0   0.0;
                 0.0  val2 0.0   0.0;
                 0.0  0.0  val2  0.0;
                 0.0  0.0  0.0   1.0 ];
    % compute the current cube after rotation
    tempOut = rot_matXY * outer_tilde;
    tempIn = rot_matXY * inner_tilde;
    tempIn(:, 1:4) = scaling1 * tempIn(:, 1:4);
    tempIn(:, 5:8) = scaling2 * tempIn(:, 5:8);
    % plot
    axis([-2 2 -2 2])
    showCube(tempOut, 'k');
    showCube(tempIn, 'k');
    connect(tempOut, tempIn, 'k')
    view(43,12)
    % save as a png file
    saveas(gcf,num2str(k),formattype);
    % clear canvas
    clf(gcf)
end


%% 3D rotation for XOY
% fixed z, rotate x and y
function rot_mat = rotMatXY(theta)
a = sind(theta);
b = cosd(theta);
rot_mat = [   b  -a 0.0 0.0;
              a   b 0.0 0.0;
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 1.0 ];
end

%% 3D rotation for YOZ
% fixed x, rotate y and z
function rot_mat = rotMatYZ(theta)
a = sind(theta);
b = cosd(theta);
rot_mat = [ 1.0 0.0 0.0 0.0;
            0.0   b  -a 0.0;
            0.0   a   b 0.0;
            0.0 0.0 0.0 1.0 ];
end

%% connect inner cube and outer cube
function connect(outer, inner, c)
hold on;
for k = 1:8
    x = [ outer(1, k) inner(1, k) ];
    y = [ outer(2, k) inner(2, k) ];
    z = [ outer(3, k) inner(3, k) ];
    plot3(x, y, z,'--', 'Color',c, 'LineWidth',2);
end
hold off
end

%%
function showCube(x, c)
% 
% This function plots the cube shape in 3-D.
% 
% Input: x: (x,y,z) coordinates as 3xM matrix c: line color
% 

hold on;

% Indices of bottom square
idx1 = [ 1 5 7 3 1 ];
plot3(x(1,idx1),x(2,idx1),x(3,idx1),'Color',c, 'Marker','o','LineWidth',2);

% Indices of top square
idx2 = [ 2 6 8 4 2 ];
plot3(x(1,idx2),x(2,idx2),x(3,idx2),'Color',c, 'Marker','o','LineWidth',2);

% Link the two squares
plot3(x(1,1:2),x(2,1:2),x(3,1:2),'Color',c, 'Marker','o','LineWidth',2);
plot3(x(1,5:6),x(2,5:6),x(3,5:6),'Color',c, 'Marker','o','LineWidth',2);
plot3(x(1,7:8),x(2,7:8),x(3,7:8),'Color',c, 'Marker','o','LineWidth',2);
plot3(x(1,3:4),x(2,3:4),x(3,3:4),'Color',c, 'Marker','o','LineWidth',2);

hold off;

return
end