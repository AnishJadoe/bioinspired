import pygame 

sensor_on_img = pygame.image.load(r"bioinspired\src\robot\images\distance_sensor_on.png")
sensor_off_img = pygame.image.load(r"bioinspired\src\robot\images\distance_sensor_off.png")
sensor_on_img = pygame.transform.scale(sensor_on_img,(30,100))
sensor_off_img = pygame.transform.scale(sensor_off_img,(30,100))

# Using blit to copy content from one surface to other
# activate the pygame library .
pygame.init()
X = 600
Y = 600
 
# create the display surface object
# of specific dimension..e(X, Y).
scrn = pygame.display.set_mode((X, Y))

status = True
angle = 45
start_pos = (300,300)
while (status):
    w,h = sensor_on_img.get_size()
    pivot_point = (300,300+0.5*h)
    pygame.draw.line(scrn, (0, 255, 0), (pivot_point[0]-20, pivot_point[1]), (pivot_point[0]+20, pivot_point[1]), 3)
    pygame.draw.line(scrn, (0, 255, 0), (pivot_point[0], pivot_point[1]-20), (pivot_point[0], pivot_point[1]+20), 3)
    pygame.draw.circle(scrn, (0, 255, 0), pivot_point, 7, 0)

    image_rect = sensor_on_img.get_rect(center=start_pos)
    pygame.draw.rect(scrn,(255,0,0),image_rect, width=1)

    offset_center_to_pivot = pygame.math.Vector2(pivot_point) - image_rect.center
    rotated_offset = offset_center_to_pivot.rotate(-angle)
    rotated_image_center = (pivot_point[0] - rotated_offset.x, pivot_point[1] - rotated_offset.y)
    rotated_image = pygame.transform.rotate(sensor_on_img, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)
    pygame.draw.rect(scrn,(255,0,0),rotated_image_rect, width=1)

  # iterate over the list of Event objects
  # that was returned by pygame.event.get() method.
    for i in pygame.event.get():
        # if event object type is QUIT
        # then quitting the pygame
        # and program both.
        if i.type == pygame.QUIT:
            status = False
    # paint screen one time
    scrn.blit(rotated_image, rotated_image_rect)
    scrn.blit(sensor_on_img, sensor_on_img.get_rect(center=start_pos))
    pygame.display.flip()
# deactivates the pygame library
pygame.quit()
