%%建立高斯金字塔
init_sigma = 1.6;% initial sigma
intvls = 3;%每层3张图像
s = intvls;%每个组将有s张图像
k = 2^(1/s);
sigma = ones(1,s+3); %每组s+3幅图像
sigma(1) = init_sigma;
sigma(2) = init_sigma*sqrt(k*k-1);
%计算每组每层高斯滤波的尺度
for i = 3:s+3
    sigma(i) = sigma(i-1)*k;%保持组与组之间的连续性
end

%调整图像大小,将图像扩大两倍
input_img = imresize(input_img,2);
%对扩大两倍的图像做一次高斯平滑
input_img = gaussian(input_img, sqrt(init_sigma^2-0.5^2*4));
%计算高斯金字塔的组数
octvs = floor(log( min(size(input_img)) )/log(2)- 2);
[limg_height,img_width] = size(input_img) ;
gauss_pyr = cell(octvs,1) ;
%设置图片大小
gimg_size = zeros(octvs, 2);
gimg_size(1,:) = [img_height,img_width];
%构建不同尺度的高斯金字塔
%1、构律分组
for i = 1 : octvs
    if (i~=1)
        gimg_size(i, :)=[round(size(gauss_pyr(i-1), 1)/2) , round(size(gauss_pyr {i-1},2)/2)];
    end
    gauss_pyr{i}= zeros( gimg_size(i,1),gimg_size (i,2),s+3 ) ;
end 
%2、对每一个分组中的分层进行操作
for i = 1 : octvs
    for j = 1 : s+3
        if(i==1 &&j==1)
            gauss_pyr {i}(:,:,j)= input_img ;
        %从前一个倍频程中的s＋1图像以八度为单位对第一个图像进行下采样
        elseif (j==1)
            %调整图像大小
            gauss_pyr {i}(:,i,j) = imresize(gauss_pyr {i-1}(:,:,s+1),0.5);
        else
            %该层是上一层的二高斯卷积结集
            gauss_pyr {i}(:,:,j) = gaussian(gauss_pyr {i}(:,:,j-1), sigma(j)) ;
        end
    end
end
%%构建高斯差分金字塔
dog_pyr = cell(octvs,1);
for i = 1:octvs
    dog_pyr {i} = zeros(gimg_size(i,1), gimg_size(i,2),s+2) ;
    for j = 1:s+2
        dog_pyr{i} (:,:,j) = gauss_pyr{i}(:,:,j+1) - gauss_pyr{i}(:,:,j);
    end
end
%低对比度筛选(取值小于0.04的极值点可以抛弃)
contr_thr = 0.04;
%检测主曲率是否在某个阈值(curv_thr)之下
curv_thr = 10;
prelim_contr_thr = 0.5*contr_thr/intvls;
for i = 1:octvs
    [height,width] = size(dog_pyr {i}(:,:,1)) ;
    %找到中心极值
    for j = 2:s+1
        dog_imgs = dog_pyr{i};
        dog_img = dog_imgs(:,:,j);
        for x = img_border+1 : height-img_border
            for y = img_border+1:width-img_border
            %阈值检测
            if(abs(dog_img(x,y)) > prelim_contr_thr)
                %极值检测
                if(isExtremum (j, x, y))
                    %极值点精确定位
                    ddata = interpLocation(dog_imgs, height, width, i,j,x, y,img_border, contr_thr, max_interp_steps);
                    if(~isempty (ddata))
                        %消除边缘效应
                        if(~isEdgeLike(dog_img, ddata. x, ddata. y, curv_thr))
                            ddata_array (ddata_index) = ddata;
                            ddata_index = ddata_index + 1;
                        end
                    end
                end
            end
            end
        end
    end
end
%极值检测
for i = 1:octvs
    [height,width] = size(dog_pyr {i}(:,:,1) ) ;
    %找到中心极值
    for j = 2:s+1
        dog_imgs = dog_pyr {i} ;
        dog_img = dog_imgs(:,:,j);
        for x = img_border+1 : height-img_border
            for y = img_border+1 : width-img_border
                %阈值检测
                if (abs(dog_img(x,y)) > prelim_contr_thr)
                    %极值检测
                        if(isExtremum(j, x,y))
                            %极值点精确定位
                            ddata = interpLocation(dog_imgs, height, width, i,j, x, y,img_border, contr_thr, max_interp_steps);
                            if(~isempty(ddata))
                                %消除边缘效应
                                if(~isEdgeLike(dog_img, ddata. x, ddata.y, curv_thr))
                                    ddata_array(ddata_index) = ddata;
                                    ddata_index = ddata_index + 1;
                                end
                            end
                        end
                end
            end
        end
    end
end

function [ flag ] = isExtremum( intvl,x, y)
    value = dog_imgs(x, y,intvl);
    block = dog_imgs(x-1:x+1,y-1 : y+1,intvl-1 :intvl+1);
    if ( value > 0 && value == max(block(:)) )
        flag = 1;
    elseif ( value == min(block(:)) )
        flag = 1;
    else
        flag = 0;
    end
end

function [ ddata ] = interpLocation( dog_imgs,height, width,octv,intvl,x, y, img_border,contr_thr, max_interp_steps )
%插值尺度空间极值的位置和尺度
global init_sigma ;
global intvls;
i = 1 ;
while (i <= max_interp_steps)
    dD= deriv3D(intvl,x, y);
    %%通过三元二次函数拟合来精确确定关健点的位置和尺度
    x_hat = - inv(hessian3D(intvl,x, y))*dD ;
    %x_hat在任意维度上的偏移量大于0.5时意味着插值中心已经偏离到它的临近点上了
    if( abs(x_hat(1)) <0.5 && abs(x_hat(2))<0.5 && abs(x_hat(3))< 0.5)
        break ;
    end
    x = x + round(x_hat(1)) ;
    y - y + round (x_hat(2) ) ;
    intvl=intvl+round(x_hat(3));
    if (intvl<2|| intvl>intvls+1 || x <= img_border || y <= img_border || x > height-img_border || y > width-img_border)
    ddata=[];
    return;
    end
    i = i+1 ;
end
if (i > max_interp_steps)
    ddata = [];
    return;
end
%%低对比度筛选
contr = dog_imgs(x, y,intvl) + 0.5*dD'*x_hat;
if ( abs(contr) < contr_thr/intvls )
    ddata = [];
    return;
end
ddata.x = x;
ddata.y = y;
ddata.octv = octv;
ddata.intvl = intvl;
ddata. x_hat = x_hat ;
ddata.scl_octv = init_sigma * power(2,(intvl+x_hat(3)-1)/intvls);
end


function [ flag ] = isEdgeLike( img,x, y,curv_thr )
%%去除边缘状的点（去除边缘相应）
%一个定义不好的高斯差分算子的极值在横跨边缘的地方有较大的主曲率,%而在垂直边缘的方向有较小的主曲率。
%DOG 算子会产生较强的边缘响应,需要剔除不稳定的边缘响应点。
center = img(x, y);
dxx = img(x,y+1) + img (x, y-1) -2*center;
dyy = img(x+1,y) + img (x-1,y) -2*center;
dxy = ( img(x+1, y+1) + img(x-1,y-1) - img(x+1,y-1) - img (x-1,y+1) )/4;
tr = dxx + dyy ;
det = dxx * dyy - dxy * dxy ;

if ( det <= 0 )
    flag = 1;
    return;
elseif ( tr^2 / det< (curv_thr + 1)^2 / curv_thr )
    flag = 0;
else
    flag = 1;
end
end


%这个函数可以得到该文件夹下的两个图像匹配点有多少对。
function [ matched ] = match( des1,des2 )
distRatio = 0.6;
des2t = des2';
n = size(des1,1);matched = zeros(1, n);
for i = 1 : n
    dotprods = des1(i,:) * des2t;%计算点积的向量
    [values,index] = sort(acos(dotprods));%取反余弦并对结果进行排序
    %检查最近邻的角度是否小于distratio时间2nd
    if (values(1) < distRatio * values(2))
        matched(i) = index(1);
    else
        matched(i) = 0 ;
    end
end
end


