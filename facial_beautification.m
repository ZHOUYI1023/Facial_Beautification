clear
project_dir = pwd;
face_path = fullfile(project_dir,'Faces','Face3.jpg');
img = imread(face_path);
%% face detection
detector = vision.CascadeObjectDetector;
bbox = detector(img);
nose = [bbox(1)+floor(0.5*bbox(3)),bbox(2)+floor(2/3*bbox(4))];
origin = [bbox(1);bbox(2)-0.3*bbox(4)];
portrait_rgb = img(origin(2):bbox(2)+1.1*bbox(4),origin(1):bbox(1)+bbox(3),:);
face_with_bbox = insertObjectAnnotation(img,'rectangle',bbox,'Face');   

%% ASM
img_gray = rgb2gray(img);
load(fullfile(project_dir,'Landmarks','Example_FindFace_Landmarks_MUCT'))    
shapeModel = buildShapeModel(allLandmarks);
load(fullfile(project_dir,'SavedModels','grayModel_MUCT'));
x = findFace(img_gray,shapeModel,grayModel,nose,'visualize',0,'facefinder','nose','layout','muct');
x = reshape(x,2,[]);
face_lower = [x(1,1:15);x(2,1:15)]-origin;
face_center = mean(face_lower,2);
face_center(2)= x(2,1)-origin(2);
face_upper = -([x(1,2:14);x(2,2:14)-15]-origin)+2*face_center;
face_outline = [face_lower,face_upper];
eyebrow_right = [x(1,16:21);x(2,16:21)]-origin;
eyebrow_left = [x(1,22:27);x(2,22:27)]-origin;
eye_right = [x(1,28:31);x(2,28:31)]-origin;
eye_left = [x(1,33:36);x(2,33:36)]-origin;
nose = [x(1,38:46);x(2,38:46)]-origin;
lip = [x(1,49:60);x(2,49:60)]-origin;

figure(1)
subplot(1,2,1)
imshow(face_with_bbox)
hold on 
plot(x(1,:),x(2,:),'r.-','LineWidth',2,'MarkerSize',8)
subplot(1,2,2)
imshow(portrait_rgb)
hold on 
plot([face_outline(1,:),face_outline(1,1)],[face_outline(2,:),face_outline(2,1)],'r.-','LineWidth',2,'MarkerSize',8)

%% mask
mask0 = poly2mask(face_outline(1,:),face_outline(2,:),size(portrait_rgb,1),size(portrait_rgb,2));
mask1 = poly2mask(eyebrow_right(1,:),eyebrow_right(2,:),size(portrait_rgb,1),size(portrait_rgb,2));
mask2 = poly2mask(eyebrow_left(1,:),eyebrow_left(2,:),size(portrait_rgb,1),size(portrait_rgb,2));
mask3 = poly2mask(eye_right(1,:),eye_right(2,:),size(portrait_rgb,1),size(portrait_rgb,2));
mask4 = poly2mask(eye_left(1,:),eye_left(2,:),size(portrait_rgb,1),size(portrait_rgb,2));
mask5 = poly2mask(lip(1,:),lip(2,:),size(portrait_rgb,1),size(portrait_rgb,2));

%% RGB to CIELAB
portrait_lab = rgb2lab(portrait_rgb);
intensity = portrait_lab(:,:,1);
intensity = uint8(intensity/100*255);
color = portrait_lab(:,:,2:3);
color = uint8(color+128);
figure(2)
subplot(1,3,1)
imshow(intensity)
subplot(1,3,2)
imshow(color(:,:,1))
subplot(1,3,3)
imshow(color(:,:,2))

%% generated mask
masked_lighting = bsxfun(@times, intensity, cast(mask0,class(intensity)));
masked_smooth = bsxfun(@times, intensity, cast(mask0&~mask3&~mask4&~mask5,class(intensity)));
masked_color = bsxfun(@times, intensity, cast(mask0&~mask3&~mask4,class(intensity)));
figure(3)
subplot(1,3,1)
imshow(masked_lighting)
subplot(1,3,2)
imshow(masked_smooth)
subplot(1,3,3)
imshow(masked_color)

%% color model
% masked_face = bsxfun(@times, portrait_lab, cast(mask0&~mask3&~mask4&~mask5,class(portrait_lab)));
% face_color_data = reshape(masked_face(:,:,2:3),[],2);
% face_color_data = reshape(face_color_data(face_color_data~=[0,0]),[],2);
% mean_color = sum(face_color_data,1)/size(face_color_data,1);
% var_color = (double(face_color_data)-mean_color)'*(double(face_color_data)-mean_color)/size(face_color_data,1);
% [V,D] = eig(var_color);
% m_distance = zeros(size(portrait_lab,1),size(portrait_lab,2));
% face_mask = logical(zeros(size(portrait_lab,1),size(portrait_lab,2)));
% for i = 1:size(portrait_lab,1)
%     for j = 1:size(portrait_lab,2)
%         m_distance(i,j) = norm(inv(D)*V'*(double(reshape(portrait_lab(i,j,1:2),[2,1])) - mean_color'));
%         if m_distance(i,j) >8
%             face_mask(i,j) = true;
%         end
%     end
% end
% se = strel('cube',5);
% erodedBW = imerode(face_mask, se);
% dilatedI = imdilate(erodedBW,se);
% se = strel('cube',5);
% erodedBW = imerode(dilatedI, se);

%% image 
[gradThresh,numIter] = imdiffuseest(intensity);
lighting = imdiffusefilt(intensity,'GradientThreshold', ...
     gradThresh,'NumberOfIterations',numIter);
detail =intensity-lighting;
figure(4)
subplot(1,3,1)
imshow(intensity)
subplot(1,3,2)
imshow(lighting)
subplot(1,3,3)
imshow(detail,[0 max(max(detail))])
%lighting_smooth = bsxfun(@times, lighting, cast(mask0&~mask3&~mask4&~mask5&logical(detail<8),class(lighting)));
%lighting_unsmooth = bsxfun(@times, intensity, cast(~(mask0&~mask3&~mask4&~mask5&logical(detail<8)),class(lighting)));
lighting_smooth = bsxfun(@times, lighting, cast(mask0,class(lighting)));
detail_smooth = bsxfun(@times, detail, cast(~(mask0&~mask1&~mask2&~mask3&~mask4&~mask5&logical(detail<8)),class(detail)));
background = bsxfun(@times, intensity, cast(~mask0,class(intensity)));
color_enhanced = bsxfun(@times, color, cast(mask0&~mask1&~mask2&~mask3&~mask4,class(color)));
color_background = bsxfun(@times, color, cast(~(mask0&~mask1&~mask2&~mask3&~mask4),class(color)));
intensity_sythesize = lighting_smooth + detail_smooth + background;
color_sythesize = 1.01*color_enhanced + color_background;

image_sythesize(:,:,1)=double(intensity_sythesize)/255*100;
image_sythesize(:,:,2:3)=double(color_sythesize)-128;
image_sythesize = lab2rgb(image_sythesize);



figure(5)
subplot(1,3,1)
imshow(intensity_sythesize)
subplot(1,3,2)
imshow(color_sythesize(:,:,1))
subplot(1,3,3)
imshow(color_sythesize(:,:,2))

figure(8)
subplot(1,2,1)
imshow(portrait_rgb)
subplot(1,2,2)
imshow(image_sythesize)


