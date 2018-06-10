function visualize(file_dir, filename, threshold, num_rows, num_cols)

if ~exist('threshold', 'var')
    threshold = 0.5;
end
if ~exist('num_rows', 'var')
    num_rows = 5;
end
if ~exist('num_cols', 'var')
    num_cols = 5;
end

load([file_dir, '/', filename], 'voxels')
num_voxels = size(voxels, 1);
num_page = num_rows * num_cols;
num_figures = ceil(num_voxels / num_page);
if ~exist([file_dir '/visualize/' filename(1:end-4)])
    mkdir([file_dir '/visualize/' filename(1:end-4)])
end
for f = 1:num_figures
    fig = figure();
    set(fig,'Color','white', 'Visible', 'off');
    set(gca,'position',[0,0,1,1],'units','normalized');
    for i = (f-1) * num_page + 1:min(f*num_page, num_voxels)
        subplot(num_rows, num_cols, i-(f-1)*num_page);
        voxel = squeeze(voxels(i,:,:,:) > threshold);
        p = patch(isosurface(voxel,0.05));
        set(p,'FaceColor','red','EdgeColor','none');
        daspect([1,1,1])
        view(3); axis tight
        camlight 
        lighting gouraud;
        axis off;
    end
    name_saved = sprintf('sample_%03d.png', f);
    saveas(fig, [file_dir '/visualize/' filename(1:end-4) '/' name_saved]);
    if f == 1
            saveas(fig, [file_dir '/visualize/' filename(1:end-4) '.png']);
    end
end