function [fig, rp] = render_face(BFM, alpha, gamma, beta, varargin )

% Render it
rp.phi = 0; %1.2
rp.rho = 0; %0.2
rp.width = 128;%182;
rp.height = 112;%160;
rp.light = [0 0]; % azimuth and elevation % 120 60
rp.material = [.5, .5, .1, 1 ]; %'dull'
rp.light_color = [1 1 1];
rp = vl_argparse(rp, varargin);

shp = BFM.shapeMU + BFM.shapePC(:,1:length(alpha))*(alpha.*BFM.shapeEV(1:length(alpha)))...
     +BFM.expMU + BFM.expPC(:,1:length(gamma))*(gamma.*BFM.expEV(1:length(gamma)));
tex = BFM.texMU' + BFM.texPC(:,1:length(beta))*(beta.*BFM.texEV(1:length(beta)));

shp = reshape(shp, [ 3 prod(size(shp))/3 ])';
tex = reshape(tex, [ 3 prod(size(tex))/3 ])';
tex = min(tex, 255);

%[x,y,z]=sphere(10);x=x(:);y=y(:);z=z(:);
%trisurf(convhull([x y z]),x*30,y*40+60,z*30-25,cos(y),'facecolor','white','EdgeColor','none','FaceLighting', 'none');
%hold on;

set(gcf, 'Renderer', 'opengl');
fig_pos = get(gcf, 'Position');
fig_pos(3) = rp.width;
fig_pos(4) = rp.height;
set(gcf, 'Position', fig_pos);
set(gcf, 'ResizeFcn', @resizeCallback);

mesh_h = trimesh(...
    BFM.faces, shp(:, 1), shp(:, 3), shp(:, 2), ...
    'EdgeColor', 'none', ...
    'FaceVertexCData', tex/255, 'FaceColor', 'interp', ...
    'FaceLighting', 'phong' ...
    );
hold off;

set(gca, ...
    'DataAspectRatio', [ 1 1 1 ], ...
    'PlotBoxAspectRatio', [ 1 1 1 ], ...
    'Units', 'pixels', ...
    'GridLineStyle', 'none', ...
    'Position', [ 0 0 fig_pos(3) fig_pos(4) ], ...
    'Visible', 'off', 'box', 'off', ...
    'Projection', 'perspective' ...
    );

set(gcf, 'Color', [ 0 0 0 ]);
view(180 + rp.phi * 180 / pi, rp.rho * 180 / pi );

material(rp.material)
h_light = camlight(rp.light(1),rp.light(2));
h_light.Color = rp.light_color;
fig = gcf;
fig.Color = 'black';
fig.InvertHardcopy = 'off';

%% ------------------------------------------------------------CALLBACK--------
function resizeCallback (obj, eventdata)

fig = gcbf;
fig_pos = get(fig, 'Position');

axis = findobj(get(fig, 'Children'), 'Tag', 'Axis.Head');
set(axis, 'Position', [ 0 0 fig_pos(3) fig_pos(4) ]);

